"""Stage 03: three scored axes + ICI + exactly 115 original features."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Sources"))
spec = importlib.util.spec_from_file_location("stage03", ROOT / "Sources/03_integrated_dataset.py")
stage03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage03)


class TargetFormulaTests(unittest.TestCase):
    def test_proportional_denton_matches_annual_benchmarks(self):
        indicator = np.array([8., 9., 10., 11., 10., 12., 13., 14.])
        annual = np.array([3.25, 7.50])
        q = stage03.proportional_denton(indicator, annual)
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(q.reshape(2, 4).mean(axis=1), annual, atol=1e-8)

    def test_proportional_denton_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):
            stage03.proportional_denton(np.array([1., 2., 3., 0.]), np.array([2.]))

    def test_split_frozen_denton_preserves_benchmarks_and_prevents_label_leakage(self):
        # Synthetic 2005-2025 annual history: Train 2005-2019,
        # Validation 2020-2022 and Test 2023-2025.
        indicator = np.array([
            (100. + 2.5 * year) * seasonal
            for year in range(21) for seasonal in (.88, 1.02, 1.09, 1.01)
        ], dtype=float)
        annual = np.array([
            42. + 0.6 * year + 3 * np.sin(year) for year in range(21)
        ])
        actual = stage03.denton_with_frozen_split_boundaries(indicator, annual)
        self.assertEqual(len(actual), 84)
        np.testing.assert_allclose(
            actual.reshape(-1, 4).mean(axis=1), annual, atol=1e-9)

        # Direct Train-only solution is EXACTLY the Train portion of the
        # production code; no 2020-2025 annual GOS or VA can enter it.
        train_only = stage03.proportional_denton(indicator[:60], annual[:15])
        np.testing.assert_array_equal(actual[:60], train_only)

        val_only = stage03.proportional_denton(
            indicator[60:72], annual[15:18],
            previous_ratio=actual[59] / indicator[59])
        np.testing.assert_array_equal(actual[60:72], val_only)
        test_only = stage03.proportional_denton(
            indicator[72:], annual[18:],
            previous_ratio=actual[71] / indicator[71])
        np.testing.assert_array_equal(actual[72:], test_only)

        future_x, future_y = indicator.copy(), annual.copy()
        future_x[60:] *= 7.0
        future_y[15:] *= 11.0
        changed = stage03.denton_with_frozen_split_boundaries(future_x, future_y)
        np.testing.assert_array_equal(changed[:60], actual[:60])

        future_x2, future_y2 = indicator.copy(), annual.copy()
        future_x2[72:] *= 3.0
        future_y2[18:] *= 8.0
        changed_test = stage03.denton_with_frozen_split_boundaries(
            future_x2, future_y2)
        np.testing.assert_array_equal(changed_test[:72], actual[:72])

    def test_score_reference_is_frozen_after_full_train(self):
        train_q = pd.period_range(*stage03.SPLIT_PERIODS["train"], freq="Q").astype(str).tolist()
        q = train_q + ["2020Q1", "2023Q1"]
        values = np.tile([1., 3.], (len(train_q) + 2, 1)).astype(float)
        values[1] = [2., 5.]
        values[-2] = [1000., 6.]
        values[-1] = [-500., 0.]
        scores, bounds = stage03.score_with_train_reference(values, q, "growth")
        # Bounds fitted once from Train ONLY: min=1, max=5; no clipping.
        self.assertAlmostEqual(scores[0, 0], 0.)
        self.assertAlmostEqual(scores[0, 1], 50.)
        self.assertAlmostEqual(scores[1, 0], 25.)
        self.assertAlmostEqual(scores[1, 1], 100.)
        self.assertAlmostEqual(scores[-2, 0], 100. * 999. / 4.)
        self.assertAlmostEqual(scores[-2, 1], 125.)
        self.assertAlmostEqual(scores[-1, 0], -100. * 501. / 4.)
        self.assertAlmostEqual(scores[-1, 1], -25.)
        self.assertEqual({b["reference_start"] for b in bounds}, {train_q[0]})
        self.assertEqual({b["reference_end"] for b in bounds}, {train_q[-1]})
        self.assertEqual({b["reference_rows"] for b in bounds}, {len(train_q) * 2})
        self.assertEqual({b["max"] for b in bounds}, {5.})
        self.assertEqual({b["min"] for b in bounds}, {1.})
        self.assertTrue(all(b["fixed_train_reference"] for b in bounds))

        # Making held-out values arbitrarily extreme cannot alter Train
        # scores or the fitted reference: explicitly guard against leakage.
        changed = values.copy()
        changed[-2:] = [[1e8, -1e8], [-1e9, 1e9]]
        after, after_bounds = stage03.score_with_train_reference(changed, q, "growth")
        np.testing.assert_array_equal(after[:len(train_q)], scores[:len(train_q)])
        self.assertEqual([(b["min"], b["max"]) for b in after_bounds],
                         [(b["min"], b["max"]) for b in bounds])

    def test_score_rejects_missing_train_quarters(self):
        train_q = pd.period_range(*stage03.SPLIT_PERIODS["train"], freq="Q").astype(str).tolist()
        with self.assertRaisesRegex(ValueError, "complete Train prefix"):
            stage03.score_with_train_reference(
                np.ones((len(train_q), 2)), train_q[1:] + ["2020Q1"], "growth")
        with self.assertRaisesRegex(RuntimeError, "Degenerate TRAIN"):
            stage03.score_with_train_reference(
                np.ones((len(train_q), 2)), train_q, "growth")


@unittest.skipUnless(
    all(path.exists() for path in stage03.SPLIT_FILES.values()),
    "Requires completed raw_train.csv / raw_val.csv / raw_test.csv",
)
class ActualOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {
            name: pd.read_csv(path, low_memory=False,
                              dtype={"quarter": str, "node_key": str})
            for name, path in stage03.SPLIT_FILES.items()
        }
        cls.data = pd.concat(cls.splits.values(), ignore_index=True)
        cls.raw = pd.read_csv(stage03.RESULTS / "target_raw_components.csv",
                              dtype={"quarter": str, "node_key": str})
        cls.registry = stage03.load_nodes()
        cls.source = stage03.load_latest_sources()

    def test_exactly_four_targets_plus_115_source_features(self):
        output = self.data
        node = pd.read_csv(stage03.NODE_02, low_memory=False,
                           dtype={"quarter": str, "node_key": str})
        national = pd.read_csv(stage03.GLOBAL_02, low_memory=False,
                               dtype={"quarter": str})
        ncols = [x for x in node if x not in {"quarter", "node_key"}]
        gcols = [x for x in national if x != "quarter"]
        self.assertEqual(len(ncols), 24)
        self.assertEqual(len(gcols), 91)
        self.assertEqual(len(output), 4480)
        self.assertEqual(output.shape[1], 122)
        self.assertEqual(output.columns.tolist(), stage03.IDS + stage03.TARGETS + ncols + gcols)
        self.assertEqual(output.node_key.nunique(), 56)
        self.assertEqual(output.quarter.nunique(), 80)
        self.assertFalse(output.duplicated(["quarter", "node_key"]).any())
        self.assertTrue(output.groupby("quarter").node_key.nunique().eq(56).all())
        self.assertFalse(output[stage03.TARGETS].isna().any().any())
        self.assertTrue(np.allclose(output["target_ici"],
                                    output[stage03.TARGETS[:3]].mean(axis=1)))
        self.assertFalse(any(x.endswith(("__d1", "__d4", "__logd1", "__logd4", "_qret", "_vol"))
                             for x in output.columns))

    def test_all_joined_features_preserve_stage02_values_and_missingness(self):
        output = self.data
        industry = pd.read_csv(stage03.NODE_02, low_memory=False,
                               dtype={"quarter": str, "node_key": str})
        national = pd.read_csv(stage03.GLOBAL_02, low_memory=False,
                               dtype={"quarter": str})
        expected = output[["quarter", "node_key"]].merge(
            industry, on=["quarter", "node_key"], validate="one_to_one")
        expected = expected.merge(national, on="quarter", validate="many_to_one")
        cols = [x for x in expected if x not in {"quarter", "node_key"}]
        # CSV round-tripping has sub-ULP decimal representation differences;
        # require numeric identity to 1e-12 and exact missing-value masks.
        self.assertTrue(np.array_equal(output[cols].isna(), expected[cols].isna()))
        np.testing.assert_allclose(output[cols].to_numpy(float), expected[cols].to_numpy(float),
                                   rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_raw_axis_definitions_against_independent_official_sources(self):
        years = pd.period_range("2005Q1", "2025Q4", freq="Q").astype(str).tolist()
        quarters = pd.period_range("2006Q1", "2025Q4", freq="Q").astype(str).tolist()
        nodes = self.registry.node_key.tolist()
        raw_index = self.raw.set_index(["quarter", "node_key"])
        real_va = stage03.bea_panel(self.source, "real_va", "quarterly", years, nodes)
        real_go = stage03.bea_panel(self.source, "real_go", "quarterly", years, nodes)
        current_go = stage03.bea_panel(self.source, "current_go", "quarterly", years, nodes)
        employees = stage03.employment_panel(self.source, years, nodes)
        actual_productivity = raw_index["productivity_raw"].unstack("node_key").loc[quarters, nodes]
        actual_growth = raw_index["growth_raw"].unstack("node_key").loc[quarters, nodes]
        actual_profitability = raw_index["profitability_raw"].unstack("node_key").loc[quarters, nodes]
        expected_productivity = real_va.loc[quarters] / employees.loc[quarters]
        # Historical PRISM correction: BLS CES 523+525 denominator must match
        # the summed BEA securities + funds/trusts numerator.
        combined = self.source[
            (self.source.node_key == "funds_trusts")
            & (self.source.series == "employment_523_525_aggregate")
            & (self.source.provider == "BLS_CES")
            & (self.source.source_industry == "CEU5552300001")
        ].copy()
        self.assertEqual(combined.units.unique().tolist(), ["thousands_of_people"])
        combined["quarter"] = pd.PeriodIndex(combined.period, freq="M").asfreq("Q").astype(str)
        grouped = combined.groupby("quarter").value.agg(["mean", "count"]).loc[quarters]
        self.assertTrue(grouped["count"].eq(3).all())
        expected_productivity["funds_trusts"] = (
            (real_va.loc[quarters, "funds_trusts"]
             + real_va.loc[quarters, "securities"])
            / grouped["mean"]
        )
        np.testing.assert_allclose(actual_productivity,
                                   expected_productivity, rtol=1e-12)
        np.testing.assert_allclose(actual_growth,
                                   real_go.loc[quarters].to_numpy()
                                   / real_go.iloc[:-4].to_numpy() - 1, rtol=1e-12, atol=1e-12)
        annual = stage03.bea_panel(self.source, "gos", "annual",
                                   [str(i) for i in range(2006, 2026)], nodes)
        reconstructed = actual_profitability * current_go.loc[quarters]
        year_averages = reconstructed.groupby(reconstructed.index.str[:4]).mean()
        np.testing.assert_allclose(year_averages.loc[annual.index, nodes], annual,
                                   rtol=1e-10, atol=1e-5)

    def test_funds_productivity_correction_restored_and_audited(self):
        audit_file = stage03.RESULTS / "funds_trusts_productivity_correction.csv"
        self.assertTrue(audit_file.is_file())
        audit = pd.read_csv(audit_file, dtype={"quarter": str})
        self.assertEqual(len(audit), 80)
        self.assertEqual(audit.quarter.iloc[0], "2006Q1")
        self.assertEqual(audit.quarter.iloc[-1], "2025Q4")
        combined_real_va = (
            audit["bea_funds_real_va_million_2017_dollars"]
            + audit["bea_securities_real_va_million_2017_dollars"]
        )
        np.testing.assert_allclose(audit["bea_combined_real_va_million_2017_dollars"],
                                   combined_real_va, rtol=0, atol=0)
        np.testing.assert_allclose(audit["restored_matched_productivity"],
                                   combined_real_va / audit["ces_523_525_employment_thousands"],
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(audit["old_unmatched_productivity"],
                                   audit["bea_funds_real_va_million_2017_dollars"]
                                   / audit["qcew_525_employment_thousands"],
                                   rtol=1e-12, atol=1e-12)
        self.assertGreater(float(audit.loc[audit.quarter == "2014Q2",
                                          "old_unmatched_productivity"].iloc[0]), 10000)
        self.assertLess(float(audit.loc[audit.quarter == "2014Q2",
                                       "restored_matched_productivity"].iloc[0]), 1000)
        reported = self.raw[self.raw.node_key == "funds_trusts"].set_index("quarter")
        np.testing.assert_allclose(reported.loc[audit.quarter, "productivity_raw"],
                                   audit["restored_matched_productivity"],
                                   rtol=1e-12, atol=1e-12)
        summary = json.loads((stage03.RESULTS / "integration_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["funds_trusts_productivity_correction"]["source"],
                         "BLS CES CEU5552300001 (NAICS 523+525 combined, NSA)")
        self.assertEqual(summary["funds_trusts_productivity_correction"]["corrected_quarters"], 80)

    def test_ces_funds_denominator_is_complete_and_required(self):
        quarters = pd.period_range("2005Q1", "2025Q4", freq="Q").astype(str).tolist()
        actual = stage03.funds_aggregate_employment(self.source, quarters)
        self.assertEqual(len(actual), 84)
        self.assertTrue((actual > 0).all())
        source = self.source[~(
            (self.source.series == "employment_523_525_aggregate")
            & (self.source.period == "2014-06")
        )]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            stage03.funds_aggregate_employment(source, quarters)

    def test_score_and_ici_values_reproduce_from_raw_axes(self):
        node_order = self.registry.node_key.tolist()
        quarters = pd.period_range("2006Q1", "2025Q4", freq="Q").astype(str).tolist()
        train_periods = pd.period_range(
            *stage03.SPLIT_PERIODS["train"], freq="Q").astype(str).tolist()
        score = self.data.set_index(["quarter", "node_key"])
        raw = self.raw.set_index(["quarter", "node_key"])
        for component in ("profitability", "productivity", "growth"):
            matrix = raw[f"{component}_raw"].unstack("node_key").loc[quarters, node_order].to_numpy()
            expected, _ = stage03.score_with_train_reference(matrix, quarters, component)
            actual = score[f"target_{component}"].unstack("node_key").loc[quarters, node_order]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

            # INDEPENDENT reference audit: take the pooled 56 x 56 Train
            # rectangle and never involve any held-out values in min/max.
            training = matrix[:len(train_periods)]
            lo, hi = float(training.min()), float(training.max())
            for t, quarter in enumerate(quarters):
                direct = 100. * (matrix[t] - lo) / (hi - lo)
                np.testing.assert_allclose(actual.loc[quarter].to_numpy(),
                                           direct, rtol=1e-12, atol=1e-12)
            # Train endpoints are exactly 0 and 100, without test information.
            self.assertAlmostEqual(float(actual.iloc[:len(train_periods)].min().min()), 0.)
            self.assertAlmostEqual(float(actual.iloc[:len(train_periods)].max().max()), 100.)

        bounds = pd.read_csv(stage03.RESULTS / "target_score_bounds.csv")
        self.assertEqual(len(bounds), 80 * 3)
        self.assertTrue(bounds["reference_start"].eq("2006Q1").all())
        self.assertTrue(bounds["reference_end"].eq("2019Q4").all())
        self.assertTrue(bounds["fixed_train_reference"].all())
        for axis in ("profitability", "productivity", "growth"):
            part = bounds[bounds["component"] == axis].reset_index(drop=True)
            self.assertEqual(part["reference_rows"].tolist(), [56 * 56] * 80)
            self.assertEqual(part["min"].nunique(), 1)
            self.assertEqual(part["max"].nunique(), 1)
            self.assertTrue(part["quarter"].tolist() == quarters)
        # Values beyond frozen Train extrema must not be clipped in Val/Test.
        self.assertTrue((score.loc[("2020Q1", slice(None)), stage03.TARGETS[:3]]
                         .to_numpy(dtype=float).shape[0]) == 56)
        held_out = self.data[self.data.quarter > "2019Q4"]
        self.assertTrue((held_out[stage03.TARGETS[:3]].to_numpy() > 100.).any())

    def test_three_exact_named_train_validation_test_outputs(self):
        summary = json.loads((stage03.RESULTS / "integration_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["gos_benchmark_blocks"], {
            "train": ["2005", "2019"], "val": ["2020", "2022"],
            "test": ["2023", "2025"],
        })
        self.assertEqual(summary["target_count"], 4)
        self.assertEqual(summary["total_feature_count"], 115)
        self.assertEqual(summary["auto_generated_feature_count"], 0)
        self.assertTrue(summary["no_imputation_or_feature_selection_in_stage_03"])
        self.assertEqual(summary["target_normalization_fit_period"], ["2006Q1", "2019Q4"])
        self.assertFalse(summary["target_normalization_clipping"])
        self.assertEqual({key: path.name for key, path in stage03.SPLIT_FILES.items()}, {
            "train": "raw_train.csv", "val": "raw_val.csv", "test": "raw_test.csv"
        })
        expected = {
            "train": ("2006Q1", "2019Q4", 56, 3136),
            "val": ("2020Q1", "2022Q4", 12, 672),
            "test": ("2023Q1", "2025Q4", 12, 672),
        }
        seen = set()
        for name, (first, last, num_quarters, num_rows) in expected.items():
            frame = self.splits[name]
            self.assertEqual((frame.quarter.iloc[0], frame.quarter.iloc[-1]), (first, last))
            self.assertEqual(frame.quarter.nunique(), num_quarters)
            self.assertEqual(frame.node_key.nunique(), 56)
            self.assertEqual(len(frame), num_rows)
            self.assertEqual(len(frame.columns), 122)
            self.assertFalse(frame.duplicated(["quarter", "node_key"]).any())
            self.assertTrue(frame.groupby("quarter").node_key.nunique().eq(56).all())
            self.assertEqual(frame.columns.tolist(), self.data.columns.tolist())
            self.assertFalse(seen.intersection(frame.quarter.unique()))
            seen.update(frame.quarter.unique())
            self.assertEqual(summary["split_periods"][name], [first, last])
            self.assertEqual(summary["split_rows"][name], num_rows)
            self.assertEqual(summary["outputs"][name],
                             f"Data/03_integrated_dataset/raw_{name}.csv")
        self.assertEqual(len(seen), 80)
        self.assertEqual(sum(len(frame) for frame in self.splits.values()), 4480)
        self.assertEqual(summary["split_ratio_quarters"],
                         {"train": 0.7, "val": 0.15, "test": 0.15})
        self.assertFalse((stage03.DATA / "prism_integrated_quarterly.csv").exists())

    def test_split_rejects_duplicate_and_missing_period(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate industry-quarter"):
            stage03.split_integrated_panel(pd.concat(
                [self.data, self.data.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(RuntimeError, "exact 2006Q1"):
            stage03.split_integrated_panel(self.data[self.data.quarter != "2006Q1"])

    def test_quarterly_mean_line_chart_and_csv(self):
        """The plotted series is the current quarter's 56-industry mean."""
        from PIL import Image

        csv_path = stage03.RESULTS / "target_quarterly_mean_timeseries.csv"
        png_path = stage03.RESULTS / "target_quarterly_mean_timeseries.png"
        self.assertTrue(csv_path.is_file())
        self.assertTrue(png_path.is_file())
        means = pd.read_csv(csv_path, dtype={"quarter": str})
        expected_quarters = pd.period_range("2006Q1", "2025Q4", freq="Q").astype(str).tolist()
        self.assertEqual(means.columns.tolist(), ["quarter"] + stage03.TARGETS)
        self.assertEqual(means.quarter.tolist(), expected_quarters)
        independent = (self.data.groupby("quarter", sort=True)[stage03.TARGETS]
                       .agg(["mean", "count"]))
        self.assertTrue(independent.xs("count", axis=1, level=1).eq(56).all().all())
        expected_means = independent.xs("mean", axis=1, level=1).loc[expected_quarters]
        np.testing.assert_allclose(means[stage03.TARGETS].to_numpy(),
                                   expected_means[stage03.TARGETS].to_numpy(),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(means["target_ici"],
                                   means[stage03.TARGETS[:3]].mean(axis=1),
                                   rtol=1e-12, atol=1e-12)
        with Image.open(png_path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreaterEqual(img.width, 1800)
            self.assertGreaterEqual(img.height, 800)
        report = json.loads((stage03.RESULTS / "integration_summary.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["target_mean_timeseries"]["quarters"], 80)
        self.assertEqual(report["target_mean_timeseries"]["industries_per_quarter"], 56)
        self.assertEqual(report["target_mean_timeseries"]["plot"],
                         "Results/03_integrated_dataset/target_quarterly_mean_timeseries.png")

    def test_quarterly_mean_chart_rejects_incomplete_node_cross_section(self):
        with self.assertRaisesRegex(RuntimeError, "56 industries"):
            stage03.save_target_mean_timeseries(self.data.iloc[1:].copy())


if __name__ == "__main__":
    unittest.main()
