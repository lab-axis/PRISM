"""New canonical Stage04: all 3 Target axes use economic-identity priors."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"Sources/04_gnn_benchmark.py"
spec = importlib.util.spec_from_file_location("prism04_three_axis_test", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ThreeAxisIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.panel = m.load_panel(ROOT)
        cls.num,cls.den,cls.meta = m.economic_components(cls.panel)
        cls.alpha,cls.audit = m.select_train_structural_priors(
            cls.panel,cls.num,cls.den,cls.meta)

    def test_structural_inputs_match_stage03_past_only_history(self):
        self.assertLess(self.meta["past_profitability_input_max_abs"],1e-9)
        self.assertLess(self.meta["unchanged_productivity_growth_max_abs"],1e-9)
        scores=self.num/self.den
        scores[...,2]-=1.
        low=np.asarray(self.meta["fixed_train_min"])
        high=np.asarray(self.meta["fixed_train_max"])
        scores=100.*(scores-low)/(high-low)
        np.testing.assert_allclose(
            scores[:-1,:,0],self.panel.x[1:,:,0],atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose(
            scores[...,1:],self.panel.y[...,1:],atol=1e-9,rtol=1e-10)
        self.assertEqual(self.num.shape,(68,59,3))
        self.assertEqual(self.den.shape,(68,59,3))
        self.assertTrue((self.den>0).all())
        self.assertEqual(len(self.meta["target_definitions"]),3)

    def test_all_axes_use_frozen_full_train_robust_alpha(self):
        self.assertEqual(self.alpha.shape,(3,))
        np.testing.assert_array_equal(self.alpha,m.STAGE04_FIXED_ALPHA)
        self.assertEqual(len(self.audit),len(m.STAGE04_ACTIVE_ALPHA_GRID))
        self.assertEqual(m.STAGE04_FIXED_ALPHA,(0.25,0.1,0.25))
        self.assertTrue(self.audit.audit_train_end.eq("2019Q4").all())
        self.assertTrue(self.audit.alpha_decision_quarters.eq(56).all())
        self.assertTrue(self.audit.alpha_decision_years.eq(14).all())
        for j,axis in enumerate(
                ("profitability","productivity","growth")):
            eligible=self.audit.loc[
                self.audit[f"{axis}_within_paired_one_se"],"coefficient"]
            self.assertAlmostEqual(float(self.alpha[j]),float(eligible.min()))
            self.assertEqual(int(self.audit[f"{axis}_is_frozen_alpha"].sum()),1)
        data=m.component_structural_prior(
            self.num,self.den,self.meta,np.arange(12,68),self.alpha)
        self.assertEqual(data.shape,(56,59,3))
        self.assertTrue(np.isfinite(data).all())
        self.assertFalse(np.allclose(
            data[44:],self.panel.y[55:67]))

    def test_future_components_never_used_at_one_forecast_origin(self):
        origins=np.array([15,40,56,61])
        original=m.component_structural_prior(
            self.num,self.den,self.meta,origins,self.alpha)
        for i,t in enumerate(origins):
            nn=self.num.copy()
            dd=self.den.copy()
            # Current and all later economically unknown components cannot
            # change a prediction at t. Growth denominator t is known.
            nn[t:]+=200_000.
            dd[t:,:,:2]*=4.
            after=m.component_structural_prior(
                nn,dd,self.meta,np.array([t]),self.alpha)
            np.testing.assert_allclose(original[i:i+1],after,rtol=0,atol=0)
            nn=self.num.copy()
            nn[t-1,:,1]+=100.
            moved=m.component_structural_prior(
                nn,self.den,self.meta,np.array([t]),self.alpha)
            self.assertFalse(np.array_equal(
                original[i:i+1,:,1],moved[...,1]))
            # Profitability is intentionally NOT taken from the static
            # numerator: the same-origin Stage03 lag is authoritative.
            info=dict(self.meta)
            info["profitability_asof_lag1"]=self.meta[
                "profitability_asof_lag1"].copy()
            info["profitability_asof_lag1"][t]+=100.
            changed=m.component_structural_prior(
                self.num,self.den,info,np.array([t]),self.alpha)
            self.assertFalse(np.array_equal(
                original[i:i+1,:,0],changed[...,0]))

    def test_next_quarter_inputs_have_safe_previous_score(self):
        all_priors=np.zeros((68,59,3))
        all_priors[12:]=m.component_structural_prior(
            self.num,self.den,self.meta,np.arange(12,68),self.alpha)
        fold=m.prepare_three_axis_fold(self.panel,all_priors,56,56,68)
        np.testing.assert_allclose(
            fold["val_last"]*fold["y_scale"]+fold["y_center"],
            self.panel.x[56:68,:,:3],atol=3e-5,rtol=3e-5)
        self.assertGreater(np.abs(
            self.panel.x[56:68,:,0]-self.panel.y[55:67,:,0]).mean(),0.)
        self.assertEqual(fold["train_y"].shape,(44,1,59,3))
        self.assertEqual(fold["val_y"].shape,(12,1,59,3))
        self.assertEqual(fold["val_structure"].shape,(12,59,3))
        self.assertEqual(fold["val_x"].shape,(12,12,59,157))

    def test_one_model_factory_supports_all_gnn_architectures(self):
        centers=np.zeros(157,dtype=np.float32)
        scale=np.ones(157,dtype=np.float32)
        yc=np.zeros(3,dtype=np.float32)
        ys=np.ones(3,dtype=np.float32)
        feature=torch.zeros((1,12,59,157))
        mask=torch.zeros_like(feature)
        past=torch.zeros((1,59,3))
        structure=torch.ones((1,59,3))
        context=torch.zeros((1,59,9))
        for name in m.GNN_MODELS:
            with self.subTest(model=name):
                net=m.ThreeAxisStructureGNN(name,centers,scale,yc,ys)
                out=net(feature,mask,torch.eye(59),past,structure,context)
                self.assertEqual(tuple(out.shape),(1,1,59,3))
                self.assertTrue(torch.isfinite(out).all())
                loss=m.three_axis_weighted_huber(
                    out,torch.zeros_like(out),past,torch.ones(3))
                self.assertTrue(torch.isfinite(loss))

    def test_test_file_not_read(self):
        seen=[]
        old=pd.read_csv
        def checked(file,*args,**kwargs):
            basename=Path(file).name
            if basename=="raw_test.csv":
                self.fail("Stage04 opened Test")
            seen.append(basename)
            return old(file,*args,**kwargs)
        with patch.object(pd,"read_csv",side_effect=checked):
            panel=m.load_panel(ROOT)
            m.economic_components(panel)
        self.assertIn("raw_train.csv",seen)
        self.assertIn("raw_val.csv",seen)
        self.assertNotIn("raw_test.csv",seen)


if __name__=="__main__":
    unittest.main()
