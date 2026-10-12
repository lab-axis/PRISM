"""Direct vs residual vs economic-prior methods: SAME original Stage03 Y[t]."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/04_gnn_benchmark.py"
spec = importlib.util.spec_from_file_location("prism04_output_tests",SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ThreeOutputModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.panel=m.load_panel(ROOT)
        n,d,i=m.economic_components(cls.panel)
        alpha,_=m.select_train_structural_priors(cls.panel,n,d,i)
        struct=np.empty_like(cls.panel.y)
        struct[:12]=cls.panel.y[:12]
        struct[12:]=m.component_structural_prior(
            n,d,i,np.arange(12,68),alpha)
        cls.fold=m.prepare_three_axis_fold(cls.panel,struct,48,48,56)

    def test_three_modes_share_one_parameterized_network_and_labels(self):
        self.assertEqual(m.STAGE04_OUTPUT_MODES,
                         ("direct","residual","structural"))
        fold=self.fold
        options=[]
        for mode in m.STAGE04_OUTPUT_MODES:
            m.seed_all(2026)
            net=m.ThreeAxisStructureGNN(
                "B-MTGNN",fold["x_center"],fold["x_scale"],
                fold["y_center"],fold["y_scale"],prediction_mode=mode)
            net.eval()
            options.append(net)
        reference=options[0].state_dict()
        for model in options[1:]:
            for name,v in model.state_dict().items():
                torch.testing.assert_close(v,reference[name],rtol=0,atol=0)
        x=torch.as_tensor(fold["val_x"][:1])
        mask=torch.as_tensor(fold["val_mask"][:1])
        last=torch.as_tensor(fold["val_last"][:1])
        prior=torch.as_tensor(fold["val_structure"][:1])
        context=torch.as_tensor(fold["val_context"][:1])
        with torch.inference_mode():
            direct,residual,structural=[
                model(x,mask,torch.eye(59),last,prior,context)[:,0]
                for model in options]
        torch.testing.assert_close(residual-direct,last,rtol=0,atol=2e-5)
        torch.testing.assert_close(structural-direct,prior,rtol=0,atol=2e-5)
        self.assertTrue(np.array_equal(
            fold["val_y"],np.stack([fold["val_y"] for _ in
                                     m.STAGE04_OUTPUT_MODES])[0]))

    def test_all_gnns_accept_three_absolute_output_formulations(self):
        f=self.fold
        x=torch.as_tensor(f["val_x"][:1])
        mask=torch.as_tensor(f["val_mask"][:1])
        last=torch.as_tensor(f["val_last"][:1])
        prior=torch.as_tensor(f["val_structure"][:1])
        context=torch.as_tensor(f["val_context"][:1])
        for name in m.GNN_MODELS:
            for mode in m.STAGE04_OUTPUT_MODES:
                with self.subTest(model=name,mode=mode):
                    net=m.ThreeAxisStructureGNN(
                        name,f["x_center"],f["x_scale"],
                        f["y_center"],f["y_scale"],
                        prediction_mode=mode)
                    out=net(x,mask,torch.eye(59),last,prior,context)
                    self.assertEqual(tuple(out.shape),(1,1,59,3))
                    self.assertTrue(torch.isfinite(out).all())

    def test_all_temporal_neural_baselines_accept_59_nodes(self):
        f=self.fold
        x=torch.as_tensor(f["val_x"][:1])
        mask=torch.as_tensor(f["val_mask"][:1])
        last=torch.as_tensor(f["val_last"][:1])
        prior=torch.as_tensor(f["val_structure"][:1])
        context=torch.as_tensor(f["val_context"][:1])
        with torch.inference_mode():
            for name in (*m.NEURAL_BASELINES,*m.EXTRA_NEURAL_BASELINES):
                with self.subTest(model=name):
                    net=m.ThreeAxisStructureGNN(
                        name,f["x_center"],f["x_scale"],
                        f["y_center"],f["y_scale"],
                        nodes=59,prediction_mode="structural")
                    net.eval()
                    result=net(x,mask,torch.eye(59),last,prior,context)
                    self.assertEqual(tuple(result.shape),(1,1,59,3))
                    self.assertTrue(torch.isfinite(result).all())

    def test_same_train_only_epoch_and_target_contract_for_all_modes(self):
        f=self.fold
        self.assertEqual(tuple(f["train_y"].shape),(36,1,59,3))
        self.assertEqual(tuple(f["val_y"].shape),(8,1,59,3))
        self.assertEqual(f["train_origins"].tolist(),list(range(12,48)))
        self.assertEqual(f["val_origins"].tolist(),list(range(48,56)))
        for mode in m.STAGE04_OUTPUT_MODES:
            with self.subTest(mode=mode):
                model=m.ThreeAxisStructureGNN(
                    "GCN-GRU",f["x_center"],f["x_scale"],
                    f["y_center"],f["y_scale"],prediction_mode=mode)
                z={k:torch.as_tensor(f[k][:1]) for k in (
                    "train_x","train_mask","train_last","train_y",
                    "train_context","train_structure")}
                p=model(z["train_x"],z["train_mask"],torch.eye(59),
                        z["train_last"],z["train_structure"],z["train_context"])
                cost=m.three_axis_weighted_huber(
                    p,z["train_y"],z["train_last"],
                    torch.as_tensor(f["y_scale"]))
                self.assertTrue(torch.isfinite(cost))
                cost.backward()
                self.assertIsNotNone(
                    model.change_head[-1].weight.grad)


class FullGridContracts(unittest.TestCase):
    def test_three_mode_grid_keeps_models_with_one_fixed_policy_each(self):
        registry = m.full_baseline_registry()
        jobs = [m.full_job_grid(mode, registry)
                for mode in m.STAGE04_OUTPUT_MODES]
        self.assertEqual(len({len(items) for items in jobs}), 1)
        expected=254
        self.assertEqual(len(jobs[0]),expected)
        self.assertEqual(sum(map(len,jobs)),3*expected)
        for items in jobs:
            self.assertEqual(sum(m._full_original_job(job, registry)
                                 for job in items), 227)
            self.assertEqual(
                {job[1] for job in items if job[1] in m.GNN_MODELS},
                set(m.GNN_MODELS))
            policies={}
            for _,model,scaler,missing,seed in items:
                self.assertEqual(
                    (scaler,missing),m.fixed_preprocessing_policy(model))
                policies.setdefault(model,set()).add((scaler,missing))
            self.assertTrue(all(len(values)==1 for values in policies.values()))
            self.assertFalse(any(s=="robust" or miss=="zero"
                                 for _,_,s,miss,_ in items))
        self.assertEqual(m.full_grid_paths()["root"],
                         ROOT / "Results/04_gnn_benchmark")


if __name__ == "__main__":
    unittest.main()
