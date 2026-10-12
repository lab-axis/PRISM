"""Stage 04 only consumes Stage 03 features; no model-specific tuning."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prism04_input_contract", ROOT / "Sources/04_gnn_benchmark.py")
stage04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage04
SPEC.loader.exec_module(stage04)


class InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = stage04.load_panel(ROOT)

    def test_only_stage03_features_and_correct_order(self):
        p = self.panel
        self.assertEqual(len(p.nodes), 59)
        self.assertEqual(p.nodes,sorted(pd.read_csv(
            ROOT/"Results/01_target_data_collection/node_registry.csv"
        ).node_key.tolist()))
        self.assertEqual(p.x.shape, (68, 59, 157))
        self.assertEqual(p.y.shape, (68, 59, 3))
        self.assertEqual(p.features[:3], [
            "history_profitability", "history_productivity", "history_growth"])
        self.assertEqual(len(p.features[3:-115]), 39)
        self.assertEqual(len(p.features[-115:]), 115)
        self.assertEqual(p.features[3], "targethist_profitability_lag02")
        np.testing.assert_allclose(
            p.x[1:, :, 1:3], p.y[:-1, :, 1:3],
            rtol=1e-12, atol=1e-12)
        self.assertGreater(np.abs(p.x[1:, :, 0]-p.y[:-1, :, 0]).mean(), 0.)
        # Stage03 correctly restores 2005 profitability/productivity, so
        # those are available at the first modeled 2006Q1 origin.
        self.assertTrue(np.isfinite(p.x[0, :, :2]).all())
        self.assertTrue(np.isnan(p.x[0, :, 2]).all())
        # Previous year's annual GOS can be incorporated at a new year's
        # forecast origin, so lag02 need NOT equal a stale earlier snapshot.
        self.assertGreater(
            np.max(np.abs(p.x[56, :, 3] - p.x[55, :, 0])), 1e-6)
        np.testing.assert_array_equal(
            stage04.origin_lag_scores(p,np.array([56]),2)[0,:,0],
            p.x[56,:,3])
        for depth in (2,4,5):
            values=stage04.origin_lag_scores(p,np.arange(56,68),depth)
            self.assertEqual(values.shape,(12,59,3))
            self.assertTrue(np.isfinite(values).all())

    def test_same_origin_context_uses_stage03_revised_annual_gos(self):
        p=self.panel
        prepared=stage04.prepare(p,"standard","mask")
        c=prepared["val_context"]*prepared["y_scale"][[0,1,2,0,1,2,0,1,2]]
        # Current t knows the last two scores as updated at origin t.
        last=stage04.past_scores(p,np.arange(56,68))
        lag2=stage04.origin_lag_scores(p,np.arange(56,68),2)
        lag5=stage04.origin_lag_scores(p,np.arange(56,68),5)
        np.testing.assert_allclose(c[:,:,:3]+prepared["y_center"],last,atol=2e-5)
        np.testing.assert_allclose(c[:,:,3:6],last-lag2,atol=2e-5)
        np.testing.assert_allclose(c[:,:,6:],last-lag5,atol=2e-5)
        # At 2020Q1 the 2019 annual-GOS benchmark has become legal.
        np.testing.assert_allclose(
            p.x[56,:,0],p.y[55,:,0],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(
            p.x[56,:,3],p.y[54,:,0],rtol=1e-12,atol=1e-12)

    def test_04_never_reads_test(self):
        observed = []
        original = pd.read_csv

        def track(file, *args, **kwargs):
            observed.append(Path(file).name)
            return original(file, *args, **kwargs)

        with patch.object(pd, "read_csv", side_effect=track):
            stage04.load_panel(ROOT)
        self.assertEqual(observed, ["raw_train.csv", "raw_val.csv"])

    def test_59_node_graph_and_ml_one_hot_include_all_new_information_industries(self):
        p=self.panel
        required={
            "information_publishing","information_motion_picture",
            "information_broadcast_telecom","information_data_services"}
        self.assertTrue(required.issubset(p.nodes))
        graph=stage04.train_graph(p)
        self.assertEqual(graph.shape,(59,59))
        np.testing.assert_array_equal(graph,graph.T)
        np.testing.assert_array_equal(np.diag(graph),np.ones(59,dtype=np.float32))
        for name in required:
            self.assertGreater(int((graph[p.nodes.index(name)]>0).sum()),1)

        empty_prior=np.zeros_like(p.y)
        fold=stage04.prepare_three_axis_fold(p,empty_prior,48,48,56)
        self.assertEqual(fold["val_context"].shape,(8,59,9))
        standard_features=stage04.LOOKBACK*157
        for mode in stage04.STAGE04_OUTPUT_MODES:
            matrix=stage04._ml_design_matrix(
                p,fold,"val","Ridge","mask",mode)
            extra_prior=3 if mode=="structural" else 0
            self.assertEqual(matrix.shape,(
                8*59,2*standard_features+9+extra_prior+59))
            self.assertTrue(np.isfinite(matrix).all())
            np.testing.assert_array_equal(
                matrix[:59,-59:],np.eye(59,dtype=np.float32))

    def test_no_target_labels_concatenated_to_feature_tensor(self):
        panel = self.panel
        prepared = stage04.prepare(panel, "standard", "mask")
        self.assertEqual(stage04.HORIZON, 1)
        self.assertEqual(stage04.LOOKBACK, 12)
        self.assertEqual(prepared["train_x"].shape, (44, 12, 59, 157))
        self.assertEqual(prepared["val_x"].shape, (12, 12, 59, 157))
        self.assertEqual(prepared["x_center"].shape, (157,))
        self.assertEqual(prepared["train_y"].shape, (44, 1, 59, 3))
        self.assertEqual(prepared["val_y"].shape, (12, 1, 59, 3))
        self.assertEqual(prepared["train_last"].shape, (44, 59, 3))
        self.assertEqual(prepared["val_last"].shape, (12, 59, 3))
        self.assertEqual(prepared["origins"].tolist(), list(range(12, 56)))
        self.assertEqual(prepared["val_origins"].tolist(), list(range(56, 68)))
        np.testing.assert_allclose(
            prepared["train_last"] * prepared["y_scale"] + prepared["y_center"],
            stage04.past_scores(panel,prepared["origins"]), rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(
            prepared["val_last"] * prepared["y_scale"] + prepared["y_center"],
            stage04.past_scores(panel,prepared["val_origins"]), rtol=1e-6, atol=1e-5)
        # Current label values may change; input tensors must not.
        changed = stage04.Panel(panel.x.copy(), panel.y.copy(), panel.quarters,
                                panel.nodes, panel.names, panel.features,
                                panel.train_count, panel.hashes)
        changed.y[:] += 200
        after = stage04.prepare(changed, "standard", "mask")
        # Floating-point reductions in the normalizer can differ at ~1e-17
        # between separate calls; this is not access to the changed labels.
        np.testing.assert_allclose(prepared["train_x"], after["train_x"],
                                   rtol=0, atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(prepared["val_x"], after["val_x"],
                                   rtol=0, atol=1e-7, equal_nan=True)

    def test_persistence_consumes_stage03_past_only_scores(self):
        p = self.panel
        np.testing.assert_array_equal(stage04.baseline_prediction(p, "Persistence")[0],
                                      p.x[56, :, :3])
        np.testing.assert_array_equal(stage04.baseline_prediction(p, "Persistence"),
                                      p.x[56:68, :, :3])
        np.testing.assert_array_equal(stage04.baseline_prediction(p, "SeasonalNaive")[:4],
                                      stage04.origin_lag_scores(
                                          p, np.arange(56,60), 4))

    def test_current_quarter_target_is_not_seen_until_next_origin(self):
        p = self.panel
        initial = stage04.prepare(p, "standard", "mask")
        altered = stage04.Panel(
            p.x.copy(), p.y.copy(), p.quarters, p.nodes, p.names,
            p.features, p.train_count, p.hashes)
        # Poisoning a held-out Target without changing its legitimate lagged
        # source must not modify any model input, including the next anchor.
        altered.y[56] += np.array([500., 100., 700.])
        modified = stage04.prepare(altered, "standard", "mask")
        np.testing.assert_array_equal(initial["val_x"][0], modified["val_x"][0])
        np.testing.assert_array_equal(
            initial["val_mask"][0], modified["val_mask"][0])
        np.testing.assert_array_equal(
            initial["val_last"][0], modified["val_last"][0])
        self.assertFalse(np.array_equal(
            initial["val_y"][0], modified["val_y"][0]))
        np.testing.assert_array_equal(initial["val_last"][1], modified["val_last"][1])
        np.testing.assert_array_equal(
            initial["train_last"], modified["train_last"])
        # By contrast, a real change to Stage03's next-quarter lagged feature
        # should affect only forecasts made AFTER that observation.
        altered.x[57, :, :3] += 100.
        changed_lag = stage04.prepare(altered, "standard", "mask")
        np.testing.assert_array_equal(initial["val_last"][0], changed_lag["val_last"][0])
        self.assertFalse(np.array_equal(initial["val_last"][1], changed_lag["val_last"][1]))

    def test_one_fixed_training_protocol_and_registry(self):
        self.assertEqual(len(stage04.GNN_MODELS), 41)
        self.assertEqual(len(stage04.BASELINES), 6)
        self.assertEqual(len(stage04.DEFAULT_SEEDS), 5)
        self.assertEqual(stage04.FIXED_LEARNING_RATE, .001)
        self.assertEqual(stage04.FIXED_BOOSTING["n_estimators"], 60)
        self.assertFalse(hasattr(stage04, "COMMON_RECIPES"))
        self.assertFalse(hasattr(stage04, "BOOSTING_RECIPES"))
        self.assertEqual(
            sum(stage04._full_original_job(job,stage04.full_baseline_registry())
                for job in stage04.full_job_grid(
                    "direct",stage04.full_baseline_registry())),227)
        model = stage04.ForecastModel("GCN-GRU", 157, 8, 8, 12, 1)
        self.assertEqual(model.values.in_features, 157)
        self.assertFalse(hasattr(model, "gain"))
        self.assertFalse(hasattr(model, "skip"))
        x = torch.randn(1, 12, 8, 157, requires_grad=True)
        pred = model(x, torch.zeros_like(x), torch.eye(8), torch.randn(1, 8, 3))
        self.assertEqual(pred.shape, (1, 1, 8, 3))
        pred.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
