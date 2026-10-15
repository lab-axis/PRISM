"""PRISM 04: B-MTGNN-only, ten seeds, target formulation and robustness."""
from __future__ import annotations
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from pipeline_support import require_current_results

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "prism04_bmtgnn_dev",ROOT/"Sources/04_bmtgnn_benchmark.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


class BMTGNNDevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.panel=m.load_panel(ROOT)
        cls.economic,_=m.make_fold(cls.panel)

    def test_only_one_model_and_ten_seeds(self):
        self.assertEqual(m.GNN_MODELS,("B-MTGNN",))
        self.assertEqual(m.MODELS,("B-MTGNN",))
        self.assertEqual(m.DEFAULT_SEEDS,tuple(range(2026,2036)))
        self.assertEqual(m.STAGE04_OUTPUT_MODES,
                         ("direct","residual","structural"))
        jobs=m.benchmark_jobs()
        self.assertEqual(len(jobs),30)
        self.assertEqual(len(set(jobs)),30)
        for mode in m.STAGE04_OUTPUT_MODES:
            self.assertEqual(
                [seed for md,_,seed in jobs if md==mode],
                list(m.DEFAULT_SEEDS))

    def test_no_other_estimator_or_other_split_access(self):
        self.assertFalse(hasattr(m,"run_test"))
        self.assertEqual(m.MODELS,("B-MTGNN",))
        self.assertEqual(m.fixed_preprocessing_policy("B-MTGNN"),
                         ("standard","mask"))
        self.assertNotIn(
            "raw_test.csv",
            (ROOT/"Sources/04_bmtgnn_benchmark.py").read_text(encoding="utf-8"))

    def test_train_dev_inputs_and_targets(self):
        self.assertEqual(self.panel.x.shape,(68,59,157))
        self.assertEqual(self.panel.y.shape,(68,59,3))
        self.assertEqual(self.panel.quarters[56],"2020Q1")
        self.assertEqual(self.panel.quarters[-1],"2022Q4")
        self.assertEqual(self.economic.shape,(68,59,3))
        fold=m.prepare_three_axis_fold(
            self.panel,self.economic,56,56,68)
        self.assertEqual(fold["train_x"].shape,(44,12,59,157))
        self.assertEqual(fold["val_x"].shape,(12,12,59,157))
        self.assertEqual(fold["train_origins"].tolist(),list(range(12,56)))

    def test_dev_labels_never_enter_model_inputs(self):
        fold=m.prepare_three_axis_fold(
            self.panel,self.economic,56,56,68)
        modified=replace(self.panel,y=self.panel.y.copy())
        modified.y[56:]+=12345.
        altered=m.prepare_three_axis_fold(
            modified,self.economic,56,56,68)
        for field in ("val_x","val_mask","val_last","val_context",
                      "val_structure","train_x","train_y"):
            np.testing.assert_array_equal(fold[field],altered[field])
        self.assertFalse(np.array_equal(fold["val_y"],altered["val_y"]))

    def test_method_difference_is_only_anchor(self):
        fold=m.prepare_three_axis_fold(
            self.panel,self.economic,56,56,68)
        models=[]
        for mode in m.STAGE04_OUTPUT_MODES:
            m.seed_all(2026)
            net=m.ThreeAxisStructureGNN(
                "B-MTGNN",fold["x_center"],fold["x_scale"],
                fold["y_center"],fold["y_scale"],
                prediction_mode=mode)
            net.eval()
            models.append(net)
        for key,weights in models[0].state_dict().items():
            for other in models[1:]:
                torch.testing.assert_close(
                    weights,other.state_dict()[key],atol=0,rtol=0)
        z={key:torch.as_tensor(fold[f"val_{key}"][:1])
           for key in ("x","mask","last","structure","context")}
        with torch.inference_mode():
            values=[
                model(z["x"],z["mask"],torch.eye(59),
                      z["last"],z["structure"],z["context"])[:,0]
                for model in models]
        torch.testing.assert_close(
            values[1]-values[0],z["last"],rtol=0,atol=2e-5)
        torch.testing.assert_close(
            values[2]-values[0],z["structure"],rtol=0,atol=2e-5)

    def test_mc_dropout_is_reproducible_and_not_zero(self):
        fold=m.prepare_three_axis_fold(
            self.panel,self.economic,56,56,68)
        inputs={
            f"val_{key}":torch.as_tensor(fold[f"val_{key}"][:1])
            for key in ("x","mask","last","structure","context")}
        self.assertNotIn("val_y",inputs)
        m.seed_all(2026)
        model=m.ThreeAxisStructureGNN(
            "B-MTGNN",fold["x_center"],fold["x_scale"],
            fold["y_center"],fold["y_scale"])
        with m.isolated_snapshot_inference(model):
            a=m.mtgnn_bayesian_inference(
                model,inputs,torch.eye(59),"B-MTGNN",2026,
                fold["y_center"],fold["y_scale"])
        with m.isolated_snapshot_inference(model):
            b=m.mtgnn_bayesian_inference(
                model,inputs,torch.eye(59),"B-MTGNN",2026,
                fold["y_center"],fold["y_scale"])
        self.assertEqual(a["inference_draws"],30)
        self.assertGreater(float(a["predictive_mc_std"].mean()),0.)
        np.testing.assert_array_equal(a["prediction"],b["prediction"])
        np.testing.assert_array_equal(
            a["predictive_mc_var"],a["predictive_mc_std"]**2)

    def test_frozen_results_are_exact_thirty_jobs(self):
        require_current_results(4)
        root=m.RESULTS
        status=m._read_json(root/"run_manifest.json")
        self.assertEqual(status["status"],"DEV_FROZEN")
        self.assertEqual(status["completed_jobs"],30)
        self.assertEqual(status["total_jobs"],30)
        self.assertEqual(status["contract"],m.benchmark_contract())
        rows=m._read_completed_jobs(status["training_source_sha256"])
        self.assertEqual(len(rows),30)
        self.assertEqual(
            {row["model"] for row in rows},{"B-MTGNN"})
        for folder in ("scores","predictions","models"):
            self.assertEqual(len(list((root/folder).iterdir())),30)
        detail=pd.read_csv(root/"dev_scores.csv")
        summary=pd.read_csv(root/"dev_summary.csv")
        epoch=pd.read_csv(root/"dev_bmtgnn_epoch_stability.csv")
        self.assertEqual(len(detail),30)
        self.assertEqual(len(epoch),90)
        self.assertEqual(len(summary),8)
        self.assertEqual(set(summary.model),
                         {"B-MTGNN","EconomicPrior",*m.NAIVE_MODELS})
        self.assertEqual(m.run_development()["status"],"DEV_FROZEN")

    def test_frozen_method_follows_dev_only_rule(self):
        frame=pd.DataFrame([
            dict(model="B-MTGNN",mode=mode,
                 mae_mean=mae,ici_mae_mean=ici)
            for mode,mae,ici in (
                ("direct",2.0,1.3),("residual",1.6,1.4),
                ("structural",1.6,1.2))])
        self.assertEqual(m.select_bmtgnn_mode(frame),"structural")
        require_current_results(4)
        locked=m._read_json(m.RESULTS/"dev_selection.json")
        self.assertEqual(locked["seed_count"],10)
        summary=pd.read_csv(m.RESULTS/"dev_summary.csv")
        self.assertEqual(locked["selected_mode"],m.select_bmtgnn_mode(summary))

    def test_epoch_predictions_and_mc_uncertainty_are_saved(self):
        require_current_results(4)
        target=self.panel.y[56:68]
        for mode in m.STAGE04_OUTPUT_MODES:
            for seed in m.DEFAULT_SEEDS:
                with self.subTest(mode=mode,seed=seed):
                    _,path,_=m._paths(mode,"B-MTGNN",seed)
                    with np.load(path,allow_pickle=False) as data:
                        self.assertEqual(
                            tuple(data["epochs"]),m.STAGE04_ROBUST_SNAPSHOTS)
                        self.assertEqual(data["epoch_predictions"].shape,
                                         (3,12,59,3))
                        self.assertEqual(data["mc_std"].shape,(12,59,3))
                        np.testing.assert_array_equal(
                            data["epoch_predictions"][1],data["prediction"])
                        self.assertTrue(np.isfinite(data["prediction"]).all())
                        row=m._read_json(m._paths(mode,"B-MTGNN",seed)[0])
                        self.assertAlmostEqual(
                            np.abs(data["prediction"]-target).mean(),
                            row["axis_mae"],places=8)

    def test_summarization_from_only_saved_bmtgnn(self):
        require_current_results(4)
        locked=m.summarize_development()
        self.assertIn(locked["selected_mode"],m.STAGE04_OUTPUT_MODES)
        rows=pd.read_csv(m.RESULTS/"dev_summary.csv")
        self.assertEqual(set(rows.model),
                         {"B-MTGNN","EconomicPrior",*m.NAIVE_MODELS})

    def test_robustness_and_paired_method_scores_match_original_predictions(self):
        """Every reported year, seed and epoch comes from the original 3 axes."""
        require_current_results(4)
        truth=self.panel.y[56:68]
        detail=pd.read_csv(m.RESULTS/"dev_scores.csv").set_index(
            ["mode","seed"])
        epochs=pd.read_csv(
            m.RESULTS/"dev_bmtgnn_epoch_stability.csv").set_index(
                ["mode","seed","epoch"])
        summaries=pd.read_csv(m.RESULTS/"dev_summary.csv").set_index(
            ["model","mode"])
        for mode in m.STAGE04_OUTPUT_MODES:
            for seed in m.DEFAULT_SEEDS:
                with self.subTest(mode=mode,seed=seed):
                    _,path,_=m._paths(mode,"B-MTGNN",seed)
                    with np.load(path,allow_pickle=False) as bank:
                        for i,epoch in enumerate(m.STAGE04_ROBUST_SNAPSHOTS):
                            original=m.score_forecast(
                                bank["epoch_predictions"][i],truth)
                            reported=epochs.loc[(mode,seed,epoch)]
                            self.assertAlmostEqual(
                                original["axis_mae"],reported.axis_mae,
                                places=10)
                            self.assertAlmostEqual(
                                original["ici_mae"],reported.ici_mae,
                                places=10)
                            if epoch==75:
                                row=detail.loc[(mode,seed)]
                                for year in range(3):
                                    self.assertAlmostEqual(
                                        original[f"year_{year}_mae"],
                                        row[f"year_{year}_mae"],places=10)
                                    self.assertAlmostEqual(
                                        original[f"year_{year}_ici_mae"],
                                        row[f"year_{year}_ici_mae"],
                                        places=10)
        for mode in m.STAGE04_OUTPUT_MODES:
            scores=detail.loc[mode]
            row=summaries.loc[("B-MTGNN",mode)]
            self.assertAlmostEqual(
                scores.axis_mae.mean(),row.mae_mean,places=10)
            self.assertAlmostEqual(
                scores.axis_mae.std(ddof=1),row.mae_seed_sd,places=10)
            self.assertAlmostEqual(
                scores.ici_mae.mean(),row.ici_mae_mean,places=10)
        # Matched-seed comparisons never treat nodes or quarters as
        # independent repetitions of the training algorithm.
        structural=detail.loc["structural"].sort_index()
        residual=detail.loc["residual"].sort_index()
        self.assertEqual(structural.index.tolist(),residual.index.tolist())
        difference=structural.axis_mae-residual.axis_mae
        self.assertEqual(len(difference),10)
        self.assertTrue(np.isfinite(difference.to_numpy()).all())

    def test_original_layout_and_added_robustness_plots(self):
        require_current_results(4)
        from PIL import Image
        originals={
            "01_three_modes_model_selection.png",
            "02_actual_vs_modes_with_persistence.png",
            "03_change_vs_modes_with_persistence.png",
            "04_gnn_ensemble_k_by_mode.png"}
        for mode in ("Direct","Residual","Structural"):
            originals.update({
                f"{mode}/01_model_comparison.png",
                f"{mode}/02_ensemble_selection.png",
                f"{mode}/03_actual_vs_predictions_with_persistence.png",
                f"{mode}/04_axis_ici_comparison.png"})
        additions={
            *(f"Robustness/{i:02d}_{name}.png" for i,name in enumerate((
                "seed_distribution","seed_epoch_heatmap",
                "worst_year_seed_regret","industry_error_dispersion",
                "mc_uncertainty_error_relation","year_seed_stability"),1)),
            *(f"Methodology/{i:02d}_{name}.png" for i,name in enumerate((
                "target_axis_methods","annual_methods",
                "paired_seed_methods","quarterly_errors"),1))}
        generated=set(m.plot_development())
        self.assertEqual(generated,originals|additions)
        folder=m.RESULTS/"plots"
        actual={p.relative_to(folder).as_posix()
                for p in folder.rglob("*.png")}
        self.assertEqual(actual,originals|additions)
        for name in generated:
            with Image.open(folder/name) as pic:
                pic.verify()


if __name__=="__main__":
    unittest.main()
