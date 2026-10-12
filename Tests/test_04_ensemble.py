"""Stage 04 ensemble: original robust 1-SE rule, GNN-only candidates."""

import importlib.util
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prism04_ensemble_test", ROOT / "Sources/04_gnn_benchmark.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class SelectionMethodTests(unittest.TestCase):
    def test_anticorrelated_errors_select_two_without_fixed_k(self):
        # Error vectors are opposite and their equal-weight combination is
        # perfect. No arbitrary minimum/maximum ensemble size is imposed.
        pattern = np.broadcast_to(np.arange(1, 7)[None, :, None],
                                  (12, 6, 3)).astype(float)
        y = np.zeros((12, 6, 3))
        bank = {
            "A": np.repeat(pattern[None], 5, axis=0),
            "B": np.repeat((-pattern)[None], 5, axis=0),
            "C": np.repeat((1.5 * pattern)[None], 5, axis=0),
        }
        path, audit, corr, selected = m.ensemble_selection_path(bank, y)
        self.assertEqual(selected, ["A", "B"])
        self.assertEqual(int(path.selected_k.iloc[0]), 2)
        self.assertEqual(len(path), 3)
        self.assertAlmostEqual(float(path.loc[path.k == 2, "axis_mae"].iloc[0]), 0.)
        self.assertAlmostEqual(float(corr.loc["A", "B"]), -1.)
        self.assertEqual(int(audit.chosen.sum()), 3)
        self.assertEqual(len(audit), 6)

    def test_redundant_errors_select_single_member(self):
        y = np.zeros((12, 6, 3))
        v = np.arange(6, dtype=float)[None, :, None] + np.zeros_like(y)
        models = {name: np.repeat(v[None], 5, axis=0) for name in ("A", "B", "C")}
        path, _, corr, selected = m.ensemble_selection_path(models, y)
        self.assertEqual(selected, ["A"])
        self.assertEqual(int(path.global_min_k.iloc[0]), 1)
        self.assertEqual(int(path.selected_k.iloc[0]), 1)
        np.testing.assert_allclose(corr.to_numpy(), np.ones((3, 3)))

    def test_invalid_prediction_or_outcome_rejected(self):
        y = np.zeros((12, 6, 3))
        with self.assertRaisesRegex(ValueError, "at least one model"):
            m.ensemble_selection_path({}, y)
        with self.assertRaisesRegex(ValueError, "invalid five-seed"):
            m.ensemble_selection_path({"A": np.zeros((1, 12, 6, 3))}, y)
        with self.assertRaisesRegex(ValueError, "finite Validation truth"):
            m.ensemble_selection_path({"A": np.zeros((5, 12, 6, 3))},
                                      np.full_like(y, np.nan))

    def test_ranking_matches_the_displayed_seed_mean_forecast(self):
        y = np.zeros((12, 6, 3))
        a = np.stack([np.full_like(y, 16. if i % 2 == 0 else -16.)
                      for i in range(5)])
        b = np.stack([np.full_like(y, 8.) for _ in range(5)])
        path, _, _, selected = m.ensemble_selection_path({"A": a, "B": b}, y)
        # Averaging 5 forecasts BEFORE computing the actual plotted MAE
        # yields A=3.2 (better than B=8), despite its 5 separately scored
        # forecasts averaging MAE=16. The old code incorrectly picked B.
        self.assertEqual(path.added_model.iloc[0], "A")
        self.assertAlmostEqual(float(path.axis_mae.iloc[0]), 3.2)
        self.assertAlmostEqual(
            float(path.axis_mae_separately_scored_seed_mean.iloc[0]), 16.)
        self.assertEqual(selected, ["A"])
        self.assertEqual(len(path), 2)
        self.assertEqual(len(selected), int(path.selected_k.iloc[0]))


class ExportContractTests(unittest.TestCase):
    def test_one_canonical_stage04_result_directory(self):
        out = ROOT / "Results/04_gnn_benchmark"
        self.assertEqual(m.full_grid_paths()["root"], out)
        self.assertFalse((out / "three_mode_full").exists())

    def test_paused_run_can_read_migrated_predictions(self):
        out = ROOT / "Results/04_gnn_benchmark"
        manifest_path = out / "run_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("No benchmark run yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest["signature"]
        if signature["source_sha256"] != hashlib.sha256(
                (ROOT/"Sources/04_gnn_benchmark.py").read_bytes()).hexdigest():
            self.skipTest("Previous preprocessing experiment is not the fixed-policy run")
        expected=3*len(m.full_job_grid(
            "direct",m.full_baseline_registry()))
        self.assertEqual(manifest["planned_runs"],expected)
        for mode in m.STAGE04_OUTPUT_MODES:
            tag = m._full_job_id((mode, "B-MTGNN", "standard", "mask", 2026))
            if not (out / "metrics" / f"{tag}.json").exists():
                self.skipTest(f"Not yet cached: {tag}")
            row, prediction = m._full_cache_record(
                (mode, "B-MTGNN", "standard", "mask", 2026),
                signature, m.full_grid_paths())
            self.assertEqual(row["status"], "PASS")
            self.assertEqual(prediction.shape, (12, 59, 3))
            self.assertEqual(row["inference_regime"],"B_MTGNN_MC_DROPOUT")
            self.assertEqual(row["inference_draws"],30)
            with np.load(out/"predictions"/f"{tag}.npz") as saved:
                self.assertIn("predictive_mc_std",saved.files)
                self.assertTrue((saved["predictive_mc_std"]>0).any())
            import torch
            checkpoint=torch.load(
                out/"checkpoints"/f"{tag}.pt",
                map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["model"],"B-MTGNN")
            self.assertIn("model_state_dict",checkpoint)
            self.assertNotIn("predictive_mc_std",checkpoint)
            self.assertNotIn("predictive_mc_var",checkpoint)

    def test_final_lock_replays_from_current_three_mode_experiment(self):
        out = ROOT / "Results/04_gnn_benchmark"
        published = out / "selected_mode_and_ensemble.json"
        if not published.is_file():
            self.skipTest("Full 3-mode benchmark not finished yet")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["signature"]["source_sha256"] != hashlib.sha256(
                (ROOT/"Sources/04_gnn_benchmark.py").read_bytes()).hexdigest():
            self.skipTest("Finished benchmark belongs to a previous 04 source version")
        report = json.loads(published.read_text())
        lock = json.loads((out / "selected_ensemble_pool.json").read_text())
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["completed_runs"],3*len(m.full_job_grid(
            "direct",m.full_baseline_registry())))
        self.assertEqual(lock["status"], "LOCKED")
        self.assertEqual(lock["stage"], "04")
        self.assertEqual(lock["model_family"], "GNN_ONLY")
        self.assertEqual(lock["prediction_mode"], report["selected_mode"])
        self.assertFalse(lock["test_opened"])
        self.assertFalse(lock["target_definition_modified"])
        self.assertEqual(lock["model_count"], len(lock["models"]))
        self.assertEqual(len(set(lock["models"])), len(lock["models"]))
        self.assertTrue(set(lock["models"]).issubset(set(m.GNN_MODELS)))
        self.assertEqual(lock["seeds"], m.DEFAULT_SEEDS)
        self.assertAlmostEqual(sum(lock["equal_weights"]), 1.)
        self.assertEqual(len(list((out / "plots").glob("*.png"))), 4)
        for mode in ("Direct", "Residual", "Structural"):
            self.assertFalse((out / mode).exists())
            self.assertEqual(len(list((out / "plots" / mode).glob("*.png"))), 4)
            for name in ("01_model_comparison.png",
                         "02_ensemble_selection.png",
                         "03_actual_vs_predictions_with_persistence.png",
                         "04_axis_ici_comparison.png"):
                self.assertGreater((out / "plots" / mode / name).stat().st_size,1000)

        with np.load(out / "ensemble_predictions.npz") as values:
            seed_predictions = values["seed_predictions"]
            actual = values["prediction"]
            truth = values["truth"]
            self.assertEqual(values["selected_models"].tolist(), lock["models"])
        self.assertEqual(seed_predictions.shape, (5, 12, 59, 3))
        np.testing.assert_allclose(actual, seed_predictions.mean(0))
        self.assertAlmostEqual(float(np.abs(actual - truth).mean()),
                               lock["selected_mae"])
        self.assertAlmostEqual(float(np.abs((actual - truth).mean(-1)).mean()),
                               lock["selected_ici_mae"])

        replay = []
        for name in lock["models"]:
            policy = lock["input_policies"][name]
            predictions = []
            for seed in lock["seeds"]:
                tag = m._full_job_id((
                    lock["prediction_mode"], name, policy["scaler"],
                    policy["missing"], seed))
                with np.load(out / "predictions" / f"{tag}.npz") as data:
                    predictions.append(data["prediction"])
            replay.append(np.stack(predictions))
        np.testing.assert_allclose(seed_predictions, np.mean(replay, axis=0),
                                   atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
