"""Stage 03: three scored axes + ICI + exactly 115 original features."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        cls.target_registry = pd.read_csv(
            stage03.RESULTS / "target_feature_registry.csv")
        cls.target_features = cls.target_registry.feature.tolist()
        cls.registry = stage03.load_nodes()
        cls.source = stage03.load_latest_sources()
        cls.bounds = pd.read_csv(stage03.RESULTS / "target_score_bounds.csv")
        cls.history_inputs = stage03.causal_history_targets(
            cls.source, cls.data[stage03.IDS + stage03.TARGETS],
            cls.bounds, cls.registry.node_key.tolist())
        cls.prehistory = stage03.pretrain_history_inputs(
            cls.source, cls.bounds, cls.registry.node_key.tolist())
        cls.profitability_asof = stage03.asof_profitability_history(
            cls.source, cls.history_inputs, cls.prehistory,
            cls.bounds, cls.registry.node_key.tolist())
        cols = ["quarter", "node_key", *stage03.TARGETS[:3]]
        cls.extended_history = pd.concat(
            [cls.prehistory[cols], cls.history_inputs[cols]],
            ignore_index=True)

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
        self.assertEqual(len(output), 80 * stage03.NODE_COUNT)
        self.assertEqual(len(self.target_features), 42)
        self.assertEqual(output.shape[1], 3 + 4 + 42 + 115)
        self.assertEqual(output.columns.tolist(),
                         stage03.IDS + stage03.TARGETS + self.target_features + ncols + gcols)
        self.assertEqual(self.target_features[:3], stage03.HISTORY_COLUMNS)
        self.assertEqual(self.target_features[3], "targethist_profitability_lag02")
        self.assertFalse(any("ici" in c for c in self.target_features))
        self.assertEqual(output.node_key.nunique(), stage03.NODE_COUNT)
        self.assertEqual(output.quarter.nunique(), 80)
        self.assertFalse(output.duplicated(["quarter", "node_key"]).any())
        self.assertTrue(output.groupby("quarter").node_key.nunique().eq(stage03.NODE_COUNT).all())
        self.assertFalse(output[stage03.TARGETS].isna().any().any())
        self.assertTrue(np.allclose(output["target_ici"],
                                    output[stage03.TARGETS[:3]].mean(axis=1)))
        self.assertFalse(any(x.endswith(("__d1", "__d4", "__logd1", "__logd4", "_qret", "_vol"))
                             for x in output.columns))

    def test_all_four_new_information_nodes_keep_exact_native_employment_scopes(self):
        expected = {
            "information_publishing": {"CEU5051100001"},
            "information_motion_picture": {"CEU5051200001"},
            "information_broadcast_telecom": {
                "CEU5051500001", "CEU5051700001"},
            "information_data_services": {
                "CEU5051800001", "CEU5051900001"},
        }
        self.assertEqual(stage03.NODE_COUNT, 59)
        self.assertFalse(self.registry.node_key.eq("information").any())
        self.assertTrue(set(expected).issubset(set(self.registry.node_key)))
        quarters = pd.period_range("2005Q1", "2025Q4", freq="Q").astype(str).tolist()
        model_quarters = quarters[4:]
        real_va = stage03.bea_panel(self.source, "real_va", "quarterly",
                                    quarters, self.registry.node_key.tolist())
        employment = stage03.employment_panel(
            self.source, quarters, self.registry.node_key.tolist())
        real_go = stage03.bea_panel(self.source, "real_go", "quarterly",
                                    quarters, self.registry.node_key.tolist())
        raw = self.raw.set_index(["quarter", "node_key"])
        for node, codes in expected.items():
            with self.subTest(node=node):
                source = self.source.loc[
                    self.source.node_key.eq(node) &
                    self.source.series.eq("employment")]
                self.assertEqual(set(source.source_industry), codes)
                self.assertEqual(set(source.provider), {"BLS_CES"})
                self.assertEqual(set(source.units), {"thousands_of_people"})
                self.assertEqual(len(source), 276 * len(codes))
                expected_productivity = (
                    real_va.loc[model_quarters, node] /
                    employment.loc[model_quarters, node])
                observed_productivity = raw.loc[
                    (model_quarters, node), "productivity_raw"]
                np.testing.assert_allclose(
                    observed_productivity, expected_productivity,
                    rtol=1e-12, atol=1e-12)
                expected_growth = (
                    real_go.loc[model_quarters, node].to_numpy() /
                    real_go.iloc[:-4][node].to_numpy() - 1)
                np.testing.assert_allclose(
                    raw.loc[(model_quarters, node), "growth_raw"],
                    expected_growth, rtol=1e-12, atol=1e-12)

    def test_stage02_missing_one_information_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            industry = pd.read_csv(stage03.NODE_02, low_memory=False,
                                   dtype={"quarter": str, "node_key": str})
            missing = (industry.node_key.eq("information_data_services") &
                       industry.quarter.eq("2020Q1"))
            self.assertEqual(int(missing.sum()), 1)
            path = Path(directory) / "node_features.csv"
            industry.loc[~missing].to_csv(path, index=False)
            with patch.object(stage03, "NODE_02", path):
                with self.assertRaisesRegex(RuntimeError, "84 quarters and 59 industry"):
                    stage03.load_features()

    def test_pre_2006_observed_profitability_and_productivity_are_not_discarded(self):
        """2005 history is a legitimate prior observation, not a future label."""
        prior = self.prehistory.set_index(["quarter", "node_key"])
        self.assertEqual(len(prior), 4 * stage03.NODE_COUNT)
        self.assertTrue(np.isfinite(prior[["target_profitability",
                                          "target_productivity"]]).all().all())
        self.assertTrue(prior["target_growth"].isna().all())
        self.assertFalse(
            ((self.source.provider.eq("BEA")) &
             self.source.series.eq("real_go") &
             self.source.frequency.eq("quarterly") &
             self.source.period.str.startswith("2004")).any())
        nodes = self.registry.node_key.tolist()
        periods = [f"2005Q{x}" for x in range(1, 5)]
        annual_gos = stage03.bea_panel(
            self.source, "gos", "annual", ["2005"], nodes).iloc[0]
        va = stage03.bea_panel(
            self.source, "current_va", "quarterly", periods, nodes)
        go = stage03.bea_panel(
            self.source, "current_go", "quarterly", periods, nodes)
        gos_quarters = pd.DataFrame({
            node: stage03.proportional_denton(
                va[node].to_numpy(float),
                np.asarray([annual_gos[node]], dtype=float))
            for node in nodes}, index=periods)
        raw = gos_quarters / go
        bounds = self.bounds.loc[self.bounds.component.eq("profitability")].iloc[0]
        expected = 100. * (raw - bounds["min"]) / (bounds["max"] - bounds["min"])
        np.testing.assert_allclose(
            prior["target_profitability"].unstack("node_key").loc[
                periods, nodes], expected, rtol=1e-12, atol=1e-12)
        early = self.data.loc[self.data.quarter.eq("2006Q1")].set_index("node_key")
        np.testing.assert_allclose(
            early.loc[nodes, "history_profitability"],
            prior.loc[("2005Q4", nodes), "target_profitability"],
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            early.loc[nodes, "history_productivity"],
            prior.loc[("2005Q4", nodes), "target_productivity"],
            rtol=1e-12, atol=1e-12)

    def test_future_annual_benchmarks_do_not_enter_restored_2005_history(self):
        changed = self.source.copy()
        altered = (changed.provider.eq("BEA") &
                   changed.series.eq("gos") &
                   changed.frequency.eq("annual") &
                   changed.period.isin(["2006", "2019", "2022"]))
        changed.loc[altered, "value"] *= 2.
        newer = stage03.pretrain_history_inputs(
            changed, self.bounds, self.registry.node_key.tolist())
        np.testing.assert_array_equal(
            newer[stage03.TARGETS[:3]],
            self.prehistory[stage03.TARGETS[:3]])
        changed_2005 = self.source.copy()
        true_history = (changed_2005.provider.eq("BEA") &
                        changed_2005.series.eq("gos") &
                        changed_2005.frequency.eq("annual") &
                        changed_2005.period.eq("2005"))
        changed_2005.loc[true_history, "value"] *= 1.01
        legitimate = stage03.pretrain_history_inputs(
            changed_2005, self.bounds, self.registry.node_key.tolist())
        self.assertFalse(np.allclose(
            legitimate["target_profitability"],
            self.prehistory["target_profitability"], rtol=1e-10, atol=1e-10))

    def test_semantically_overlapping_native_employment_is_preserved_as_past_feature(self):
        """Employment is the productivity denominator at t, legal input for t+1."""
        periods = pd.period_range("2005Q1", "2025Q4", freq="Q").astype(str).tolist()
        nodes = self.registry.node_key.tolist()
        native = pd.read_csv(
            stage03.NODE_02, dtype={"quarter": str, "node_key": str}
        ).pivot(index="quarter", columns="node_key", values="qcew_employment")
        employed = stage03.employment_panel(self.source, periods, nodes)
        np.testing.assert_allclose(
            native.loc[periods, nodes], employed,
            rtol=1e-12, atol=1e-12)
        self.assertIn("qcew_employment", self.data)
        # Keep the native observation in its OWN period. Shift only at the
        # next-quarter model input, never delete this legitimate historical
        # productivity-denominator signal.
        published = self.data.set_index(["quarter", "node_key"])
        np.testing.assert_allclose(
            published.loc[
                ("2021Q1", nodes), "qcew_employment"].to_numpy(float),
            employed.loc["2021Q1", nodes],
            rtol=1e-12, atol=1e-12)

    def test_previous_target_scores_are_exactly_three_lagged_inputs(self):
        self.assertEqual(stage03.HISTORY_COLUMNS,
                         ["history_profitability", "history_productivity", "history_growth"])
        table = self.data.sort_values(["node_key", "quarter"], kind="stable")
        early = self.extended_history.sort_values(
            ["node_key", "quarter"], kind="stable")
        modeled = early.quarter >= stage03.MODEL_START
        for source, history in zip(stage03.TARGETS[:3], stage03.HISTORY_COLUMNS):
            if source == "target_profitability":
                continue  # An annual GOS benchmark becomes usable at year end.
            expected = early.groupby("node_key")[source].shift(1).loc[modeled]
            np.testing.assert_allclose(table[history].to_numpy(float),
                                       expected.to_numpy(float), rtol=1e-12, atol=1e-12, equal_nan=True)
        time_major = self.data.sort_values(
            ["quarter", "node_key"], kind="stable")
        np.testing.assert_allclose(
            time_major.history_profitability.to_numpy(float),
            self.profitability_asof[:, 0].reshape(-1),
            rtol=1e-12, atol=1e-12)
        for axis in ("productivity", "growth"):
            np.testing.assert_allclose(
                table.loc[table.quarter > stage03.MODEL_START,f"history_{axis}"],
                table.groupby("node_key")[f"target_{axis}"].shift(1)
                    .loc[table.quarter > stage03.MODEL_START],
                rtol=0, atol=0, equal_nan=True)
        self.assertEqual(int(self.data[stage03.HISTORY_COLUMNS].isna().sum().sum()),
                         stage03.NODE_COUNT)
        self.assertTrue(self.data.loc[self.data.quarter == "2006Q1",
                                      ["history_profitability",
                                       "history_productivity",
                                       "targethist_profitability_mean_w04"]].notna().all().all())
        self.assertTrue(self.data.loc[self.data.quarter == "2006Q1",
                                      ["history_growth",
                                       "targethist_growth_mean_w04",
                                       "targethist_growth_lag02"]].isna().all().all())
        for quarter, prior in (("2020Q1", "2019Q4"), ("2023Q1", "2022Q4")):
            current = self.data[self.data.quarter == quarter].sort_values("node_key")
            before = self.history_inputs[
                self.history_inputs.quarter == prior].sort_values("node_key")
            np.testing.assert_allclose(current[stage03.HISTORY_COLUMNS[1:]].to_numpy(float),
                                       before[stage03.TARGETS[1:3]].to_numpy(float),
                                       rtol=1e-12, atol=1e-12)
            official_prior = self.data[
                self.data.quarter.eq(prior)].sort_values("node_key")
            np.testing.assert_allclose(
                current.history_profitability.to_numpy(float),
                official_prior.target_profitability.to_numpy(float),
                rtol=1e-12, atol=1e-12)

    def test_engineered_features_match_prior_quarters_and_hand_calculations(self):
        ordered = self.data.sort_values(["node_key", "quarter"], kind="stable")
        source_values = self.extended_history.sort_values(
            ["node_key", "quarter"], kind="stable")
        modeled = source_values.quarter >= stage03.MODEL_START
        for axis in stage03.TARGET_AXES[:3]:
            if axis == "profitability":
                continue  # Separately independently reconstructed as of t.
            source = f"target_{axis}"
            for lag in (2, 4, 8, 12):
                name = f"targethist_{axis}_lag{lag:02d}"
                expected = source_values.groupby("node_key")[source].shift(lag).loc[modeled]
                np.testing.assert_allclose(ordered[name], expected, rtol=1e-12,
                                           atol=1e-12, equal_nan=True)
            previous = source_values.groupby("node_key")[source].shift(1).loc[modeled]
            before = source_values.groupby("node_key")[source].shift(5).loc[modeled]
            np.testing.assert_allclose(
                ordered[f"targethist_{axis}_change04"], previous - before,
                rtol=1e-12, atol=1e-12, equal_nan=True)
            prev_full = source_values.groupby("node_key")[source].shift(1)
            mean = prev_full.groupby(source_values.node_key).transform(
                lambda s: s.rolling(4, min_periods=4).mean()).loc[modeled]
            np.testing.assert_allclose(
                ordered[f"targethist_{axis}_mean_w04"], mean,
                rtol=1e-12, atol=1e-12, equal_nan=True)
        peers = self.data[self.data.quarter == "2020Q1"].sort_values("node_key")
        previous_peers = self.data[self.data.quarter == "2019Q4"].sort_values("node_key")
        expected_rank = previous_peers.target_growth.rank(pct=True).to_numpy(float)
        np.testing.assert_allclose(
            peers.targethist_growth_peer_rank.to_numpy(float), expected_rank,
            rtol=1e-12, atol=1e-12)
        self.assertEqual(set(self.target_registry.latest_input_quarter_offset), {-1})
        self.assertTrue(self.target_registry.causality.eq(
            "STRICT_T_MINUS_ONE_OR_EARLIER").all())
        self.assertTrue((self.target_registry.minimum_history_quarters >= 1).all())

    def test_all_42_target_features_replay_from_prior_scores_for_59_nodes(self):
        """Independent formulas for every engineered input, including peer terms."""
        quarters = pd.period_range(
            stage03.MODEL_START, stage03.MODEL_END, freq="Q").astype(str).tolist()
        full_quarters = pd.period_range(
            stage03.HISTORY_START, stage03.MODEL_END, freq="Q").astype(str).tolist()
        nodes = sorted(self.registry.node_key)
        full_index = pd.MultiIndex.from_product(
            [full_quarters, nodes], names=["quarter", "node_key"])
        aligned = self.extended_history.set_index(
            ["quarter", "node_key"]).reindex(full_index)
        index = pd.MultiIndex.from_product(
            [quarters, nodes], names=["quarter", "node_key"])
        exported = self.data.set_index(
            ["quarter", "node_key"]).reindex(index)
        axes = aligned[stage03.TARGETS[:3]].to_numpy(float).reshape(
            84, stage03.NODE_COUNT, 3)
        reconstructed = {}
        for axis_id, axis in enumerate(stage03.TARGET_AXES):
            values = axes[:, :, axis_id]

            def lag(depth):
                if axis == "profitability":
                    return self.profitability_asof[:, depth - 1]
                out = np.full_like(values, np.nan)
                out[depth:] = values[:-depth]
                return out[4:]

            recent = lag(1)
            reconstructed[f"history_{axis}"] = recent
            for depth in stage03.TARGET_LAGS:
                reconstructed[f"targethist_{axis}_lag{depth:02d}"] = lag(depth)
            for distance in stage03.TARGET_CHANGES:
                reconstructed[f"targethist_{axis}_change{distance:02d}"] = (
                    recent - lag(distance + 1))
            full_recent = np.full_like(values, np.nan)
            full_recent[1:] = values[:-1]
            full_series = pd.DataFrame(
                full_recent, index=full_quarters, columns=nodes)
            series = pd.DataFrame(recent, index=quarters, columns=nodes)
            for window in stage03.TARGET_WINDOWS:
                if axis == "profitability":
                    window_values = np.stack(
                        [lag(depth) for depth in range(1, window + 1)])
                    valid = np.isfinite(window_values).all(axis=0)
                    filled = np.where(np.isfinite(window_values), window_values, 0.)
                    means = np.where(valid, filled.mean(0), np.nan)
                    stds = np.where(valid, filled.std(0, ddof=0), np.nan)
                else:
                    rolling = full_series.rolling(window, min_periods=window)
                    means = rolling.mean().to_numpy()[4:]
                    stds = rolling.std(ddof=0).to_numpy()[4:]
                reconstructed[f"targethist_{axis}_mean_w{window:02d}"] = means
                reconstructed[f"targethist_{axis}_std_w{window:02d}"] = stds
            window = 4
            history = np.stack([lag(k) for k in range(window, 0, -1)])
            x = np.arange(window, dtype=float) - (window - 1) / 2.
            reconstructed[f"targethist_{axis}_trend_w04"] = (
                np.sum(history * x[:, None, None], axis=0) / np.sum(x ** 2))
            reconstructed[f"targethist_{axis}_peer_gap"] = (
                recent - np.mean(recent, axis=1, keepdims=True))
            reconstructed[f"targethist_{axis}_peer_rank"] = (
                series.rank(axis=1, pct=True).to_numpy())

        self.assertEqual(set(reconstructed), set(self.target_features))
        self.assertEqual(len(reconstructed), 42)
        for name, expected in reconstructed.items():
            with self.subTest(feature=name):
                np.testing.assert_allclose(
                    exported[name].to_numpy(float),
                    expected.reshape(-1),
                    # CSV round-trip can shift Welford rolling variances
                    # by ~5e-11 when the true variance is nearly zero.
                    rtol=1e-12,
                    atol=1e-10 if "_std_w" in name else 1e-12,
                    equal_nan=True)

    def test_engineering_has_strict_no_current_or_future_target_leakage(self):
        base = self.history_inputs.copy()
        original, _ = stage03.engineer_target_history(
            base, prehistory=self.prehistory,
            profitability_asof=self.profitability_asof)
        features = self.target_features
        for boundary in ("2013Q2", "2020Q1", "2023Q1"):
            perturbed = base.copy()
            future = perturbed.quarter >= boundary
            perturbed.loc[future, "target_profitability"] += 1937.0
            perturbed.loc[future, "target_productivity"] -= 4321.0
            perturbed.loc[future, "target_growth"] *= -3.0
            perturbed.loc[future, "target_ici"] = perturbed.loc[
                future, stage03.TARGETS[:3]].mean(axis=1)
            changed, _ = stage03.engineer_target_history(
                perturbed, prehistory=self.prehistory,
                profitability_asof=self.profitability_asof)
            # Poisoning Y_t and later MUST NOT modify the input features at
            # time t or any earlier quarter. The latest allowed label is t-1.
            protected = original.quarter <= boundary
            np.testing.assert_allclose(
                original.loc[protected, features].to_numpy(float),
                changed.loc[protected, features].to_numpy(float),
                rtol=0, atol=0, equal_nan=True,
            )
            self.assertTrue(
                original.loc[original.quarter >= boundary, features].compare(
                    changed.loc[changed.quarter >= boundary, features]).shape[0] > 0)
            current_only = original.quarter == boundary
            source_before = base.loc[base.quarter ==
                                     str(pd.Period(boundary, freq="Q") - 1)].sort_values("node_key")
            changed_at_boundary = changed.loc[current_only].sort_values("node_key")
            for observed, true_column in zip(stage03.HISTORY_COLUMNS, stage03.TARGETS[:3]):
                if observed == "history_profitability":
                    continue
                np.testing.assert_allclose(
                    changed_at_boundary[observed].to_numpy(float),
                    source_before[true_column].to_numpy(float),
                    rtol=0, atol=0)
            origin_index = pd.period_range(
                stage03.MODEL_START, stage03.MODEL_END, freq="Q").astype(str).tolist().index(boundary)
            np.testing.assert_allclose(
                changed_at_boundary.history_profitability.to_numpy(float),
                self.profitability_asof[origin_index, 0],
                rtol=0, atol=0)

    def test_future_annual_gos_does_not_change_prior_input_features(self):
        future_source = self.source.copy()
        change = (future_source.series.eq("gos")
                  & future_source.frequency.eq("annual")
                  & future_source.period.eq("2022"))
        self.assertGreater(int(change.sum()), 0)
        future_source.loc[change, "value"] *= 1.01
        changed_input = stage03.causal_history_targets(
            future_source, self.data[stage03.IDS + stage03.TARGETS],
            self.bounds, self.registry.node_key.tolist())
        changed_asof = stage03.asof_profitability_history(
            future_source, changed_input, self.prehistory, self.bounds,
            self.registry.node_key.tolist())
        before, _ = stage03.engineer_target_history(
            self.history_inputs, prehistory=self.prehistory,
            profitability_asof=self.profitability_asof)
        after, _ = stage03.engineer_target_history(
            changed_input, prehistory=self.prehistory,
            profitability_asof=changed_asof)
        history_names = self.target_features
        protected = before.quarter <= "2022Q4"
        np.testing.assert_allclose(
            before.loc[protected, history_names],
            after.loc[protected, history_names],
            rtol=0, atol=0, equal_nan=True)
        # At 2023Q1, all 2022 quarters and its annual GOS are legitimately past.
        changed = before.quarter.eq("2023Q1")
        self.assertFalse(np.array_equal(
            before.loc[changed, "history_profitability"],
            after.loc[changed, "history_profitability"]))
        np.testing.assert_array_equal(
            self.history_inputs[["target_productivity", "target_growth"]].to_numpy(),
            changed_input[["target_productivity", "target_growth"]].to_numpy())

    def test_completed_gos_used_at_year_boundary_not_during_same_year(self):
        """2020Q1 may use 2019 annual GOS; 2020Q2 may not use 2020 annual GOS."""
        nodes = sorted(self.registry.node_key)
        produced = self.data.set_index(["quarter", "node_key"])
        prior = produced.loc[("2019Q4", nodes), "target_profitability"].to_numpy(float)
        np.testing.assert_allclose(
            produced.loc[("2020Q1", nodes), "history_profitability"],
            prior, rtol=1e-12, atol=1e-12)

        va = stage03.bea_panel(
            self.source, "current_va", "quarterly",
            [f"2019Q{q}" for q in range(1, 5)] + ["2020Q1"], nodes)
        go = stage03.bea_panel(
            self.source, "current_go", "quarterly", ["2020Q1"], nodes)
        annual_2019 = stage03.bea_panel(
            self.source, "gos", "annual", ["2019"], nodes).iloc[0]
        prior_ratio = annual_2019 / va.loc[
            [f"2019Q{q}" for q in range(1, 5)]].mean(0)
        raw_score = va.loc["2020Q1"] * prior_ratio / go.loc["2020Q1"]
        b = self.bounds.loc[self.bounds.component.eq("profitability")].iloc[0]
        expected = 100. * (raw_score - b["min"]) / (b["max"] - b["min"])
        np.testing.assert_allclose(
            produced.loc[("2020Q2", nodes), "history_profitability"],
            expected, rtol=1e-12, atol=1e-12)

    def test_partial_val_block_denton_uses_only_completed_years(self):
        """The 2020 annual benchmark is usable in 2021Q1, but 2021/2022 are not."""
        nodes = sorted(self.registry.node_key)
        periods = pd.period_range("2005Q1", "2020Q4", freq="Q").astype(str).tolist()
        va = stage03.bea_panel(self.source, "current_va", "quarterly", periods, nodes)
        go = stage03.bea_panel(self.source, "current_go", "quarterly", ["2020Q4"], nodes)
        annual = stage03.bea_panel(
            self.source, "gos", "annual", [str(y) for y in range(2005, 2021)], nodes)
        b = self.bounds.loc[self.bounds.component.eq("profitability")].iloc[0]
        expected = []
        for node in nodes:
            train = stage03.proportional_denton(
                va.loc[:"2019Q4", node].to_numpy(float),
                annual.loc[:"2019", node].to_numpy(float))
            val = stage03.proportional_denton(
                va.loc["2020Q1":"2020Q4", node].to_numpy(float),
                annual.loc[["2020"], node].to_numpy(float),
                previous_ratio=train[-1] / va.loc["2019Q4", node])
            value = val[-1] / go.loc["2020Q4", node]
            expected.append(100. * (value - b["min"]) / (b["max"] - b["min"]))
        observed = (self.data.loc[self.data.quarter.eq("2021Q1")]
                    .set_index("node_key").loc[nodes, "history_profitability"])
        np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-9)

    def test_future_2019_and_2024_annual_gos_only_changes_next_year_inputs(self):
        """Poison future annual benchmarks and check the precise legal boundary."""
        quarters = pd.period_range(
            stage03.MODEL_START, stage03.MODEL_END, freq="Q").astype(str).tolist()
        for year in (2019, 2024):
            with self.subTest(annual_year=year):
                changed_source = self.source.copy()
                changed_year = (
                    changed_source.provider.eq("BEA") &
                    changed_source.series.eq("gos") &
                    changed_source.frequency.eq("annual") &
                    changed_source.period.eq(str(year)))
                self.assertTrue(changed_year.any())
                changed_source.loc[changed_year, "value"] *= 1.01
                provisional = stage03.causal_history_targets(
                    changed_source, self.data[stage03.IDS + stage03.TARGETS],
                    self.bounds, self.registry.node_key.tolist())
                changed = stage03.asof_profitability_history(
                    changed_source, provisional, self.prehistory, self.bounds,
                    self.registry.node_key.tolist())
                before = self.profitability_asof
                t0 = quarters.index(f"{year + 1}Q1")
                np.testing.assert_allclose(
                    changed[:t0], before[:t0], rtol=0, atol=0, equal_nan=True)
                self.assertFalse(np.allclose(
                    changed[t0, 0], before[t0, 0], rtol=1e-12, atol=1e-12))

    def test_asof_profitability_is_independent_of_node_registry_order(self):
        """Never copy another industry's GOS when model nodes are alphabetized."""
        shuffled = self.registry.node_key.tolist()[::-1]
        reordered = stage03.asof_profitability_history(
            self.source, self.history_inputs, self.prehistory,
            self.bounds, shuffled)
        np.testing.assert_array_equal(self.profitability_asof, reordered)

    def test_future_quarterly_source_cannot_change_prior_features(self):
        future_source = self.source.copy()
        change = (future_source.series.eq("current_va")
                  & future_source.frequency.eq("quarterly")
                  & future_source.period.eq("2022Q2"))
        self.assertGreater(int(change.sum()), 0)
        future_source.loc[change, "value"] *= 1.02
        revised = stage03.causal_history_targets(
            future_source, self.data[stage03.IDS + stage03.TARGETS],
            self.bounds, self.registry.node_key.tolist())
        revised_asof = stage03.asof_profitability_history(
            future_source, revised, self.prehistory, self.bounds,
            self.registry.node_key.tolist())
        before, _ = stage03.engineer_target_history(
            self.history_inputs, prehistory=self.prehistory,
            profitability_asof=self.profitability_asof)
        after, _ = stage03.engineer_target_history(
            revised, prehistory=self.prehistory,
            profitability_asof=revised_asof)
        prior = before.quarter <= "2022Q2"
        np.testing.assert_allclose(
            before.loc[prior, self.target_features],
            after.loc[prior, self.target_features],
            rtol=0, atol=0, equal_nan=True)
        next_quarter = before.quarter.eq("2022Q3")
        self.assertFalse(np.array_equal(
            before.loc[next_quarter, "history_profitability"],
            after.loc[next_quarter, "history_profitability"]))

    def test_profitability_history_is_distinct_from_retrospective_label(self):
        aligned = self.history_inputs.sort_values(
            ["quarter", "node_key"], kind="stable")
        original = self.data.sort_values(["quarter", "node_key"], kind="stable")
        np.testing.assert_allclose(
            aligned[["target_productivity", "target_growth"]].to_numpy(),
            original[["target_productivity", "target_growth"]].to_numpy(),
            rtol=0, atol=0)
        self.assertGreater(np.mean(np.abs(
            aligned.target_profitability.to_numpy()
            - original.target_profitability.to_numpy())), 0.0)
        self.assertEqual(len(self.history_inputs), 80 * stage03.NODE_COUNT)
        self.assertTrue(np.isfinite(
            self.history_inputs[stage03.TARGETS].to_numpy()).all())

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

            # INDEPENDENT reference audit: take the pooled 56 x 59 Train
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
            self.assertEqual(part["reference_rows"].tolist(),
                             [56 * stage03.NODE_COUNT] * 80)
            self.assertEqual(part["min"].nunique(), 1)
            self.assertEqual(part["max"].nunique(), 1)
            self.assertTrue(part["quarter"].tolist() == quarters)
        # Values beyond frozen Train extrema must not be clipped in Val/Test.
        self.assertTrue((score.loc[("2020Q1", slice(None)), stage03.TARGETS[:3]]
                         .to_numpy(dtype=float).shape[0]) == stage03.NODE_COUNT)
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
        self.assertEqual(summary["auto_generated_feature_count"], 42)
        self.assertEqual(summary["target_engineered_feature_count"], 42)
        self.assertEqual(summary["model_input_feature_count"], 42 + 115)
        self.assertEqual(summary["previous_target_history_columns"], stage03.HISTORY_COLUMNS)
        self.assertEqual(summary["previous_target_history_lag_quarters"], 1)
        self.assertEqual(summary["target_feature_observation_policy"].split(";")[0],
                         "Every row t uses only information through t-1")
        self.assertTrue(summary["no_imputation_or_feature_selection_in_stage_03"])
        self.assertEqual(summary["target_normalization_fit_period"], ["2006Q1", "2019Q4"])
        self.assertFalse(summary["target_normalization_clipping"])
        self.assertEqual({key: path.name for key, path in stage03.SPLIT_FILES.items()}, {
            "train": "raw_train.csv", "val": "raw_val.csv", "test": "raw_test.csv"
        })
        expected = {
            "train": ("2006Q1", "2019Q4", 56, 56 * stage03.NODE_COUNT),
            "val": ("2020Q1", "2022Q4", 12, 12 * stage03.NODE_COUNT),
            "test": ("2023Q1", "2025Q4", 12, 12 * stage03.NODE_COUNT),
        }
        seen = set()
        for name, (first, last, num_quarters, num_rows) in expected.items():
            frame = self.splits[name]
            self.assertEqual((frame.quarter.iloc[0], frame.quarter.iloc[-1]), (first, last))
            self.assertEqual(frame.quarter.nunique(), num_quarters)
            self.assertEqual(frame.node_key.nunique(), stage03.NODE_COUNT)
            self.assertEqual(len(frame), num_rows)
            self.assertEqual(len(frame.columns), 3 + 4 + 42 + 115)
            self.assertFalse(frame.duplicated(["quarter", "node_key"]).any())
            self.assertTrue(frame.groupby("quarter").node_key.nunique().eq(stage03.NODE_COUNT).all())
            self.assertEqual(frame.columns.tolist(), self.data.columns.tolist())
            self.assertFalse(seen.intersection(frame.quarter.unique()))
            seen.update(frame.quarter.unique())
            self.assertEqual(summary["split_periods"][name], [first, last])
            self.assertEqual(summary["split_rows"][name], num_rows)
            self.assertEqual(summary["outputs"][name],
                             f"Data/03_integrated_dataset/raw_{name}.csv")
        self.assertEqual(len(seen), 80)
        self.assertEqual(sum(len(frame) for frame in self.splits.values()),
                         80 * stage03.NODE_COUNT)
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
        """The plotted series is the current quarter's 59-industry mean."""
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
        self.assertTrue(independent.xs("count", axis=1, level=1).eq(stage03.NODE_COUNT).all().all())
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
        self.assertEqual(report["target_mean_timeseries"]["industries_per_quarter"],
                         stage03.NODE_COUNT)
        self.assertEqual(report["target_mean_timeseries"]["plot"],
                         "Results/03_integrated_dataset/target_quarterly_mean_timeseries.png")

    def test_quarterly_mean_chart_rejects_incomplete_node_cross_section(self):
        with self.assertRaisesRegex(RuntimeError, "59 industries"):
            stage03.save_target_mean_timeseries(self.data.iloc[1:].copy())


if __name__ == "__main__":
    unittest.main()
