"""Regression checks for the simplified latest-only PRISM collection policy."""

import importlib.util
import hashlib
import io
import json
import inspect
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Sources"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "Sources" / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


one = module("s01_latest", "01_target_data_collection.py")
two = module("s02_latest", "02_initial_feature_pool.py")


class StageOneLatest(unittest.TestCase):
    def test_portable_official_qcew_snapshot_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "qcew_monthly_employment_2003_2025.json"
            manifest = root / "source_manifest.json"
            months = pd.period_range("2003-01", "2025-12", freq="M").astype(str)
            blob = json.dumps({"22": {month: 1000. for month in months}}).encode()
            snapshot.write_bytes(blob)
            item = snapshot.relative_to(root).as_posix()
            # Patch ROOT too: source paths in manifests are relative to ROOT.
            manifest.write_text(json.dumps({
                "files": {item: {"sha256": hashlib.sha256(blob).hexdigest()}}
            }))
            with (patch.object(one, "ROOT", root),
                  patch.object(one, "QCEW_SNAPSHOT", snapshot),
                  patch.object(one, "QCEW_MANIFEST", manifest)):
                source, digest = one.load_verified_qcew_snapshot(["22"])
                self.assertEqual(len(source["22"]), 276)
                self.assertEqual(digest, hashlib.sha256(blob).hexdigest())
                snapshot.write_bytes(blob + b" ")
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    one.load_verified_qcew_snapshot(["22"])

    def test_no_runtime_git_source_restore(self):
        for module in (one, two):
            source = inspect.getsource(module)
            self.assertNotIn("subprocess", source)
            self.assertNotIn("repository_bytes(", source)
            self.assertNotIn('"git", "show"', source)
            self.assertNotIn('"HEAD:"', source)

    def test_combined_523_525_ces_is_preserved_as_original_source(self):
        dates = pd.date_range("2003-01-01", "2025-12-01", freq="MS")
        csv_data = pd.DataFrame({
            "observation_date": dates.astype(str),
            "CEU5552300001": np.arange(len(dates), dtype=float) + 800,
        }).to_csv(index=False).encode()

        class Response:
            content = csv_data
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CEU5552300001"

            def raise_for_status(self):
                return None

        with patch.object(one.requests, "get", return_value=Response()):
            combined, meta = one.collect_latest_funds_ces(None)
        self.assertEqual(len(combined), 276)
        self.assertEqual(meta["series_id"], "CEU5552300001")
        self.assertEqual(meta["origin"], "current_FRED_BLS_series")
        self.assertEqual(combined[0]["series"], "employment_523_525_aggregate")
        self.assertEqual(combined[0]["period"], "2003-01")
        self.assertEqual(combined[-1]["period"], "2025-12")
        self.assertEqual(combined[0]["units"], "thousands_of_people")

        ordinary = [dict(provider="BLS_QCEW", series="employment",
                         frequency="monthly", period="2025-12",
                         source_industry="525", value=34900., units="persons",
                         source_url="qcew", source_release_date="")]
        combined_source = one.assign_nodes([], [], ordinary, [combined[-1]])
        self.assertEqual(set(combined_source.series), {
            "employment", "employment_523_525_aggregate"})
        self.assertEqual(set(combined_source.node_key), {"funds_trusts"})
        self.assertAlmostEqual(float(combined_source.loc[
            combined_source.series == "employment", "value"].iloc[0]), 34900.)

    def test_source_cells_use_native_values_and_no_derived_target(self):
        rows = [
            ("BEA", "gos", "annual", "2025", "Utilities", 42.0),
            ("BEA", "gos", "annual", "2025", "Hospitals", 22.0),
            ("BEA", "gos", "annual", "2025", "Nursing and residential care facilities", 3.0),
        ]
        raw_bea = [dict(provider=p, series=s, frequency=f, period=t,
                        source_industry=ind, value=v, units="millions",
                        source_url="bea", source_release_date="2026-09-30")
                   for p, s, f, t, ind, v in rows]
        qcew = [dict(provider="BLS_QCEW", series="employment",
                     frequency="monthly", period="2025-01", source_industry="22",
                     value=1000.0, units="persons", source_url="bls",
                     source_release_date="")]
        rail = [dict(provider="BLS_CES", series="employment",
                     frequency="monthly", period="2025-01",
                     source_industry="tri_434820_NSA.csv", value=149.0,
                     units="thousands_of_people", source_url="bls", source_release_date="2026-01")]
        output = one.assign_nodes(raw_bea, rail, qcew)
        self.assertEqual(len(output), 5)
        hospitals = output[output.node_key == "hospitals_nursing"]
        self.assertEqual(sorted(hospitals.value), [3.0, 22.0])
        self.assertEqual(output[output.node_key == "utilities"].series.tolist().count("gos"), 1)
        self.assertFalse(any("ici" in c.lower() or "profitability" in c.lower()
                             for c in output.columns))

    def test_rail_latest_vintage_only(self):
        csv_content = "year,month,Jan_05,Feb_05,Mar_05\n2025,12,80,81,82\n2026,1,90,91,92\n"
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("tri_434820_NSA.csv", csv_content)
        with patch.object(one, "get", return_value=data.getvalue()):
            rows, meta = one.collect_latest_rail_ces(None)
        self.assertEqual([x["value"] for x in rows], [90.0, 91.0, 92.0])
        self.assertEqual(meta["latest_vintage_month"], "2026-01")

    def test_no_restore_of_old_asof_pipeline(self):
        self.assertFalse(hasattr(one, "build_asof_sources"))
        self.assertFalse(hasattr(one, "bea_first_publications"))
        self.assertFalse(hasattr(one, "versioned_calendar"))


class StageTwoLatest(unittest.TestCase):
    def test_bls_catalog_reads_verified_portable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "metadata"
            folder.mkdir()
            catalog = folder / "ce.series"
            manifest = folder / "source_manifest.json"
            blob = b"series_id\tindustry_code\nCEU0000000001\t00000000\n"
            catalog.write_bytes(blob)
            manifest.write_text(json.dumps({
                "files": {"metadata/ce.series": {
                    "sha256": hashlib.sha256(blob).hexdigest()
                }}
            }))
            with (patch.object(two, "ROOT", root),
                  patch.object(two, "METADATA_DIR", folder),
                  patch.object(two, "METADATA_MANIFEST", manifest)):
                catalog_table = two.bls_metadata("ce", "ce.series")
                self.assertEqual(catalog_table.series_id.tolist(), ["CEU0000000001"])
                catalog.write_bytes(blob + b" ")
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    two.bls_metadata("ce", "ce.series")

    def test_bls_native_source_cache_does_not_call_legacy_git(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "ces_industry.csv"
            pd.DataFrame([
                {"series_id": "CEU0000000001", "year": 2025,
                 "period": "M01", "value": 100.0}
            ]).to_csv(cache_path, index=False)

            class OfflineSession:
                def post(self, *args, **kwargs):
                    raise two.requests.ConnectionError("Simulated offline BLS API")

            with patch.object(two, "RAW_DIR", Path(directory)):
                result, meta = two.current_bls(
                    OfflineSession(), "ces_industry", {"CEU0000000001"})
                self.assertEqual(result.value.tolist(), [100.])
                self.assertEqual(meta["origin"], "portable_official_source_cache")
                with self.assertRaisesRegex(RuntimeError, "No portable"):
                    two.current_bls(
                        OfflineSession(), "ppi_industry", {"PCU999999"},
                        refresh=False)

    def test_no_legacy_feature_multiplication_functions(self):
        self.assertFalse(hasattr(two, "add_derivatives"))
        self.assertFalse(hasattr(two, "expand_features"))
        self.assertFalse(hasattr(two, "build_asof_features"))
        source = inspect.getsource(two)
        self.assertNotIn("def add_derivatives(", source)
        self.assertNotIn(".pct_change(", source)
        self.assertNotIn(".rolling(", source)
        self.assertNotIn(".shift(", source)
        self.assertNotIn("TARGET_INGREDIENT_ASOF_ONLY", source)
        self.assertNotIn("safe_lag_quarters", source)

    def test_native_feature_inventory_rejects_expansion(self):
        values = pd.DataFrame({"industry": [1., 2.]}, index=["2025Q1", "2025Q2"])
        node = {f"native_node_{i}": values for i in range(24)}
        national = {f"native_global_{i}": pd.Series([1., 2.])
                    for i in range(91)}
        registry = []
        two.register_node(node, "BLS", "NODE", registry)
        two.register_global(national, "BLS", "GLOBAL", registry)
        mapping = [{"source": "BLS CES", "feature_id": "native_node_0",
                    "node_key": "industry", "series_id": "CEU0000000001"}]
        two.validate_native_feature_inventory(node, national, mapping, registry)
        self.assertEqual(len(registry), 115)
        self.assertEqual(len({entry["feature_id"] for entry in registry}), 115)
        self.assertFalse(any(
            "base_feature" in entry or "safe_lag_quarters" in entry
            or "target_overlap_control" in entry
            for entry in registry))
        with self.assertRaisesRegex(RuntimeError, "multiple features"):
            two.validate_native_feature_inventory(
                node, national, mapping + [
                    {"source": "BLS CES", "feature_id": "native_node_1",
                     "node_key": "industry", "series_id": "CEU0000000001"}
                ], registry)
        altered_node = dict(node)
        del altered_node["native_node_23"]
        altered_node["native_node_23__d1"] = values.diff()
        with self.assertRaisesRegex(RuntimeError, "Synthetic feature expansion"):
            two.validate_native_feature_inventory(
                altered_node, national, mapping, registry)
        with self.assertRaisesRegex(RuntimeError, r"only 24\+91"):
            two.validate_native_feature_inventory(
                dict(list(node.items())[:23]), national, mapping, registry)

    def test_quarterly_source_average_needs_all_three_months(self):
        data = pd.DataFrame([
            {"series_id": "ONE", "year": 2025, "period": "M01", "value": 12.0},
            {"series_id": "ONE", "year": 2025, "period": "M02", "value": 15.0},
            {"series_id": "ONE", "year": 2025, "period": "M03", "value": 18.0},
            {"series_id": "ONE", "year": 2025, "period": "M13", "value": 9000.0},
            {"series_id": "TWO", "year": 2025, "period": "M01", "value": 1.0},
            {"series_id": "TWO", "year": 2025, "period": "M02", "value": 2.0},
        ])
        q = two.quarterly_bls(data, ["2025Q1"])
        self.assertEqual(float(q["ONE"].iloc[0]), 15.0)
        self.assertTrue(np.isnan(q["TWO"].iloc[0]))

    def test_qcew_node_units_and_rail_ces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_sources.csv.gz"
            rows = [
                {"node_key": "utilities", "series": "employment", "provider": "BLS_QCEW",
                 "period": f"2025-{month:02d}", "source_industry": "22",
                 "value": n, "units": "persons"}
                for month, n in ((1, 1000.0), (2, 3000.0), (3, 5000.0))
            ] + [
                {"node_key": "rail_transport", "series": "employment", "provider": "BLS_CES",
                 "period": f"2025-{month:02d}", "source_industry": "tri_434820_NSA.csv",
                 "value": n, "units": "thousands_of_people"}
                for month, n in ((1, 150.0), (2, 160.0), (3, 170.0))
            ]
            pd.DataFrame(rows).to_csv(path, compression="gzip", index=False)
            nodes = [{"key": "utilities", "naics_codes": "22"},
                     {"key": "rail_transport", "naics_codes": "482"}]
            with patch.object(two, "TARGET_LATEST", path):
                table = two.latest_qcew_employment(nodes, ["2025Q1"])["qcew_employment"]
            self.assertEqual(float(table.loc["2025Q1", "utilities"]), 3.0)
            self.assertEqual(float(table.loc["2025Q1", "rail_transport"]), 160.0)

    def test_one_original_source_one_feature(self):
        quarters = ["2025Q1", "2025Q2"]
        source = pd.DataFrame({"utilities": [2.0, 3.0]}, index=quarters)
        reg = []
        a = two.register_node({"qcew_employment": source}, "QCEW", "NODE", reg)
        b = two.register_global({"bls_cpi_all_items": pd.Series(
            [100.0, 101.0], index=quarters)}, "BLS", "GLOBAL", reg)
        self.assertEqual(list(a) + list(b), ["qcew_employment", "bls_cpi_all_items"])
        self.assertEqual(len(reg), 2)
        self.assertEqual(len(set(x["feature_id"] for x in reg)), 2)

    def test_one_market_close_feature_per_ticker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            pd.DataFrame({"XYZ": [101.0, 102.0]},
                         index=pd.to_datetime(["2025-10-15", "2025-12-31"])
                         ).to_csv(root / "market_closes.csv")
            with (patch.object(two, "RAW_DIR", root),
                  patch.object(two, "MARKET_TICKERS", {"XYZ": "example"})):
                result, labels = two.latest_market(["2025Q4"], False)
            self.assertEqual(list(result), ["market_example"])
            self.assertEqual(float(result["market_example"].iloc[0]), 102.0)
            self.assertEqual(len(labels), 1)

    def test_no_vintage_reconstruction(self):
        self.assertFalse(hasattr(two, "bls_filter_flat"))
        self.assertFalse(hasattr(two, "alfred_snapshots"))
        self.assertFalse(hasattr(two, "PIT_ORIGINS"))


if __name__ == "__main__":
    unittest.main()
