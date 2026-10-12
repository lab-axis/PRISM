"""Stage04 preprocessing is assigned once by model family, not tuned on Dev."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prism04_fixed_preprocessing_contract",
    ROOT / "Sources/04_gnn_benchmark.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class FixedPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = m.load_panel(ROOT)
        cls.prior = np.zeros_like(cls.panel.y)

    def test_every_family_has_exactly_one_policy(self):
        self.assertEqual(m.fixed_preprocessing_policy("B-MTGNN"),
                         ("standard", "mask"))
        for name in (*m.NEURAL_BASELINES, *m.EXTRA_NEURAL_BASELINES,
                     "Ridge"):
            with self.subTest(model=name):
                self.assertEqual(m.fixed_preprocessing_policy(name),
                                 ("standard", "mask"))
        for name in m.NATIVE_NAN_TREES:
            with self.subTest(model=name):
                self.assertEqual(m.fixed_preprocessing_policy(name),
                                 ("none", "native"))
        for name in m.NAIVE_MODELS:
            self.assertEqual(m.fixed_preprocessing_policy(name),
                             ("none", "none"))
        for name in ("ElasticNet","Lasso","BayesianRidge","Huber",
                     "KNN","SVR","RandomForest","HistGradientBoosting",
                     "GradientBoosting","XGBoost","MLPRegressor","CatBoost"):
            with self.subTest(removed_model=name):
                with self.assertRaises(ValueError):
                    m.fixed_preprocessing_policy(name)
                with self.assertRaises(ValueError):
                    m._ml_regressor(name,2026)
        with self.assertRaises(ValueError):
            m.fixed_preprocessing_policy("UnknownLearner")

    def test_no_preprocessing_search_across_three_modes(self):
        registry = m.full_baseline_registry()
        self.assertEqual(registry["extra_ml"],["Ridge","ExtraTrees"])
        self.assertEqual(registry["original_ml"],["LightGBM"])
        self.assertEqual(registry["optional_ml_installed"],[])
        self.assertEqual(registry["optional_ml_unavailable"],[])
        for mode in m.STAGE04_OUTPUT_MODES:
            jobs = m.full_job_grid(mode, registry)
            self.assertEqual(len(jobs),254)
            self.assertEqual(len(jobs), len(set(jobs)))
            for _, name, scaler, missing, seed in jobs:
                self.assertEqual((scaler, missing),
                                 m.fixed_preprocessing_policy(name))
                self.assertNotIn(scaler, ("robust",))
                self.assertNotEqual(missing, "zero")
            names = {job[1] for job in jobs}
            for name in names:
                actual = [job for job in jobs if job[1] == name]
                self.assertEqual(
                    len(actual), 1 if name in m.NAIVE_MODELS else 5)
                self.assertEqual(
                    len({(job[2], job[3]) for job in actual}), 1)

    def test_raw_zero_then_train_standard_and_separate_mask(self):
        p = self.panel
        fold = m.prepare_three_axis_fold(
            p, self.prior, 48, 48, 56, "standard", "mask")
        raw = p.x[:m.LOOKBACK]
        missing = np.isnan(raw)
        self.assertTrue(missing.any())
        self.assertTrue(np.isfinite(fold["train_x"]).all())
        np.testing.assert_array_equal(
            fold["train_mask"][0], missing.astype(np.float32))
        centers, widths = m.fit_normalizer(p.x[:48], "standard")
        np.testing.assert_array_equal(fold["x_center"], centers.astype(np.float32))
        np.testing.assert_array_equal(fold["x_scale"], widths.astype(np.float32))
        expected = ((np.where(missing, 0., raw) - centers) / widths)
        np.testing.assert_allclose(
            fold["train_x"][0], expected, rtol=1e-6, atol=1e-6)
        # A missing observation is zero in RAW space, not an imputed
        # Train mean (nor necessarily a zero after scaling).
        np.testing.assert_allclose(
            fold["train_x"][0][missing],
            np.broadcast_to((-centers / widths), raw.shape)[missing],
            rtol=1e-6, atol=1e-6)
        self.assertTrue(np.any(np.abs(fold["train_x"][0][missing]) > 1e-4))

    def test_native_tree_inputs_are_unscaled_and_retain_exact_nans(self):
        p = self.panel
        fold = m.prepare_three_axis_fold(
            p, self.prior, 48, 48, 56, "none", "native")
        self.assertTrue(np.array_equal(fold["x_center"], np.zeros(157)))
        self.assertTrue(np.array_equal(fold["x_scale"], np.ones(157)))
        self.assertFalse(fold["train_mask"].any())  # no mask to native trees
        n = 12*157
        observed = p.x[:12].transpose(1, 0, 2).reshape(59, n)
        for name in m.NATIVE_NAN_TREES:
            with self.subTest(model=name):
                x = m._ml_design_matrix(
                    p, fold, "train", name, "native", "direct")
                self.assertEqual(x.shape, ((48-12)*59, n+9+59))
                np.testing.assert_array_equal(
                    np.isnan(x[:59, :n]), np.isnan(observed))
                np.testing.assert_allclose(
                    x[:59, :n], observed.astype(np.float32),
                    rtol=1e-6, atol=1e-6, equal_nan=True)
                self.assertTrue(np.isnan(x[:59, :n]).any())
                self.assertFalse(np.isnan(x[:, n:]).any())
                anchor = m.origin_lag_scores(p, np.array([12]), 1)[0]
                lag2 = m.origin_lag_scores(p, np.array([12]), 2)[0]
                lag5 = m.origin_lag_scores(p, np.array([12]), 5)[0]
                asof_raw_context = np.concatenate(
                    (anchor, anchor-lag2, anchor-lag5), axis=-1)
                np.testing.assert_allclose(
                    x[:59, n:n+9], asof_raw_context,
                    rtol=1e-6, atol=1e-4)

    def test_only_retained_native_trees_use_raw_score_context_and_structural_anchor(self):
        p = self.panel
        for name in m.NATIVE_NAN_TREES:
            scaler, missing = m.fixed_preprocessing_policy(name)
            new_fold = m.prepare_three_axis_fold(
                p, self.prior, 48, 48, 56, scaler, missing)
            values = m._ml_design_matrix(
                p, new_fold, "val", name, missing, "structural")
            # Structural prior is an original-score zero vector here and
            # must stay zero (not a normalized negative Train mean).
            np.testing.assert_allclose(values[:, -59-3:-59], 0., atol=1e-5)

    def test_reject_all_noncanonical_preprocessing_arms(self):
        with self.assertRaisesRegex(ValueError, "fixed"):
            m.prepare(self.panel, "robust", "mask")
        with self.assertRaisesRegex(ValueError, "fixed"):
            m.prepare(self.panel, "standard", "zero")
        for policy in (("robust", "zero"), ("standard", "zero"),
                       ("robust", "mask"), ("none", "zero")):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "predefined"):
                    m.prepare_three_axis_fold(
                        self.panel, self.prior, 48, 48, 56, *policy)

    def test_native_nan_tree_regressors_accept_missing_input(self):
        # Tiny fit, not the 762-job benchmark. Check that the pinned ML
        # implementations really support the native-NaN policy.
        rng = np.random.default_rng(20261012)
        x = rng.normal(size=(64, 6))
        x[::5, 0] = np.nan
        x[::7, 3] = np.nan
        y = rng.normal(size=64)
        for name in m.NATIVE_NAN_TREES:
            with self.subTest(model=name):
                estimator = m._ml_regressor(name, 2026)
                estimator.fit(x, y)
                forecast = np.asarray(estimator.predict(x[:8]), float)
                self.assertTrue(np.isfinite(forecast).all())


if __name__ == "__main__":
    unittest.main()
