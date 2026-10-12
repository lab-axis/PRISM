"""Regression: B-MTGNN uses posterior MC dropout; MTGNN is deterministic."""

from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "prism04_bayesian_regression",ROOT/"Sources/04_gnn_benchmark.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class BayesianInferenceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        p=m.load_panel(ROOT)
        cls.panel=p
        cls.fold=m.prepare_three_axis_fold(
            p,np.zeros_like(p.y),48,48,56,"standard","mask")
        cls.graph=torch.as_tensor(m.train_graph(
            m.Panel(p.x,p.y,p.quarters,p.nodes,p.names,p.features,48,p.hashes)))
        cls.batch={
            k:torch.as_tensor(cls.fold[k][:2])
            for k in ("val_x","val_mask","val_last",
                      "val_structure","val_context")}

    def test_identical_weights_have_distinct_predictive_distributions(self):
        # Controlled comparison: exact same network weights, two inference
        # modes. MTGNN is one deterministic prediction, Bayesian MTGNN
        # samples active dropout and averages 30 stochastic predictions.
        m.seed_all(12026)
        b=m.ThreeAxisStructureGNN(
            "B-MTGNN",self.fold["x_center"],self.fold["x_scale"],
            self.fold["y_center"],self.fold["y_scale"])
        t=m.ThreeAxisStructureGNN(
            "MTGNN",self.fold["x_center"],self.fold["x_scale"],
            self.fold["y_center"],self.fold["y_scale"])
        t.load_state_dict(b.state_dict(),strict=True)
        a=m.mtgnn_bayesian_inference(
            b,self.batch,self.graph,"B-MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        d=m.mtgnn_bayesian_inference(
            t,self.batch,self.graph,"MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        self.assertEqual(a["inference_draws"],30)
        self.assertEqual(d["inference_draws"],1)
        self.assertEqual(a["inference_regime"],"B_MTGNN_MC_DROPOUT")
        self.assertEqual(d["inference_regime"],"MTGNN_DETERMINISTIC")
        self.assertTrue(b.training)
        self.assertFalse(t.training)
        self.assertEqual(a["prediction"].shape,(2,59,3))
        self.assertFalse(np.array_equal(
            a["prediction"],d["prediction"]))
        self.assertTrue((a["predictive_mc_std"]>=0).all())
        self.assertGreater(float(a["predictive_mc_std"].mean()),1e-4)
        np.testing.assert_allclose(
            a["predictive_mc_var"],a["predictive_mc_std"]**2,
            rtol=1e-11,atol=1e-12)

    def test_bayesian_draws_are_reproducible_for_fixed_seed(self):
        m.seed_all(12026)
        model=m.ThreeAxisStructureGNN(
            "B-MTGNN",self.fold["x_center"],self.fold["x_scale"],
            self.fold["y_center"],self.fold["y_scale"])
        a=m.mtgnn_bayesian_inference(
            model,self.batch,self.graph,"B-MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        b=m.mtgnn_bayesian_inference(
            model,self.batch,self.graph,"B-MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        np.testing.assert_array_equal(a["prediction"],b["prediction"])
        np.testing.assert_array_equal(
            a["predictive_mc_std"],b["predictive_mc_std"])

    def test_mtgnn_eval_is_seed_invariant(self):
        m.seed_all(12026)
        model=m.ThreeAxisStructureGNN(
            "MTGNN",self.fold["x_center"],self.fold["x_scale"],
            self.fold["y_center"],self.fold["y_scale"])
        a=m.mtgnn_bayesian_inference(
            model,self.batch,self.graph,"MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        b=m.mtgnn_bayesian_inference(
            model,self.batch,self.graph,"MTGNN",2030,
            self.fold["y_center"],self.fold["y_scale"])
        np.testing.assert_array_equal(a["prediction"],b["prediction"])
        self.assertNotIn("predictive_mc_std",a)
        self.assertNotIn("predictive_mc_std",b)

    def test_no_current_target_labels_required_to_forecast(self):
        m.seed_all(12026)
        model=m.ThreeAxisStructureGNN(
            "B-MTGNN",self.fold["x_center"],self.fold["x_scale"],
            self.fold["y_center"],self.fold["y_scale"])
        a=m.mtgnn_bayesian_inference(
            model,self.batch,self.graph,"B-MTGNN",2026,
            self.fold["y_center"],self.fold["y_scale"])
        # No val_y is present in self.batch. Same prediction can be
        # produced without current-quarter truth or future source rows.
        self.assertNotIn("val_y",self.batch)
        self.assertTrue(np.isfinite(a["prediction"]).all())


if __name__=="__main__":
    unittest.main()
