"""Regression tests for PRISM 04's fixed latest-official-data input contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prism04_causal_input_tests", ROOT / "Sources/04_gnn_benchmark.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class ForecastOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = m.load_panel(ROOT)
        cls.num, cls.den, cls.info = m.economic_components(cls.panel)
        cls.alpha, _ = m.select_train_structural_priors(
            cls.panel, cls.num, cls.den, cls.info)
        cls.structure = np.zeros_like(cls.panel.y)
        cls.structure[12:] = m.component_structural_prior(
            cls.num, cls.den, cls.info, np.arange(12, 68), cls.alpha)

    def test_three_axis_anchor_and_all_naive_baselines_are_past_only(self):
        p = self.panel
        origins = np.arange(56, 68)
        expected = p.x[origins, :, :3]
        np.testing.assert_array_equal(m.past_scores(p,origins),expected)
        np.testing.assert_array_equal(m.baseline_prediction(p,"Persistence"),expected)
        np.testing.assert_array_equal(m.naive_full_forecast(p,"Persistence"),expected)
        np.testing.assert_array_equal(
            m.naive_full_forecast(p,"SeasonalNaive"),
            m.origin_lag_scores(p,origins,4))
        np.testing.assert_allclose(
            m.naive_full_forecast(p,"LinearDrift"),
            2*p.x[origins,:,:3]-m.origin_lag_scores(p,origins,2))
        self.assertGreater(np.abs(expected[...,0]-p.y[origins-1,:,0]).mean(),0.)

    def test_perturbing_unknown_targets_never_changes_forecast_inputs(self):
        p=self.panel
        baseline=m.prepare_three_axis_fold(p,self.structure,56,56,68)
        changed=m.Panel(p.x.copy(),p.y.copy(),p.quarters,
                        p.nodes,p.names,p.features,p.train_count,p.hashes)
        changed.y[56:,:,:] += 500.
        after=m.prepare_three_axis_fold(changed,self.structure,56,56,68)
        for key in ("val_x","val_mask","val_last","val_context","val_structure",
                    "train_x","train_last","train_context","train_structure"):
            np.testing.assert_array_equal(baseline[key],after[key])
        self.assertFalse(np.array_equal(baseline["val_y"],after["val_y"]))

    def test_future_row_nonhistorical_features_do_not_reach_origin(self):
        p=self.panel
        observed=m.prepare_three_axis_fold(p,self.structure,56,56,68)
        changed=m.Panel(p.x.copy(),p.y.copy(),p.quarters,
                        p.nodes,p.names,p.features,p.train_count,p.hashes)
        # Row t contains SAFE targethist_* fields too; poison only
        # contemporaneous external fields unavailable for prediction of Y[t].
        changed.x[56, :, -115:] += 10000.
        altered=m.prepare_three_axis_fold(changed,self.structure,56,56,68)
        for key in ("val_x","val_last","val_context","val_structure"):
            np.testing.assert_array_equal(observed[key][:1],altered[key][:1])

    def test_legal_same_origin_lags_change_context_not_past_window(self):
        p=self.panel
        baseline=m.prepare_three_axis_fold(p,self.structure,56,56,68)
        changed=m.Panel(p.x.copy(),p.y.copy(),p.quarters,
                        p.nodes,p.names,p.features,p.train_count,p.hashes)
        # x[56]'s targethist profitability lag02 was built solely from
        # 2019Q3, with its now-completed 2019 annual GOS benchmark.
        index=p.features.index("targethist_profitability_lag02")
        changed.x[56,:,index] += 5.
        updated=m.prepare_three_axis_fold(changed,self.structure,56,56,68)
        np.testing.assert_array_equal(baseline["val_x"][0],updated["val_x"][0])
        np.testing.assert_array_equal(baseline["val_last"][0],updated["val_last"][0])
        self.assertFalse(np.array_equal(
            baseline["val_context"][0],updated["val_context"][0]))

    def test_structural_profitability_uses_same_origin_t_minus_two_score(self):
        t=56 # 2020Q1: 2019 annual GOS is complete, 2020 is not.
        before=m.component_structural_prior(
            self.num,self.den,self.info,np.array([t]),self.alpha)
        # Changing a superseded static 2019Q3 GOS estimate must not alter
        # the structural forecast; Stage03's t=2020Q1 lag02 is authoritative.
        wrong=self.num.copy()
        wrong[t-2,:,0]*=1.2
        after=m.component_structural_prior(
            wrong,self.den,self.info,np.array([t]),self.alpha)
        np.testing.assert_array_equal(before,after)
        correct=dict(self.info)
        correct["profitability_asof_lag2"]=self.info[
            "profitability_asof_lag2"].copy()
        correct["profitability_asof_lag2"][t] += 3.
        changed=m.component_structural_prior(
            self.num,self.den,correct,np.array([t]),self.alpha)
        self.assertGreater(
            np.abs(before[...,0]-changed[...,0]).max(),1e-5)
        np.testing.assert_array_equal(before[...,1:],changed[...,1:])

    def test_future_asof_score_rows_never_change_prior_predictions(self):
        """Even the metadata's per-origin snapshots cannot read t+1 or later."""
        for t in (12,47,48,55,56,57,60,67):
            with self.subTest(forecast_origin=t):
                before=m.component_structural_prior(
                    self.num,self.den,self.info,np.array([t]),self.alpha)
                changed=dict(self.info)
                for key in ("profitability_asof_lag1","profitability_asof_lag2"):
                    values=np.asarray(self.info[key]).copy()
                    values[t+1:]+=100000.
                    changed[key]=values
                after=m.component_structural_prior(
                    self.num,self.den,changed,np.array([t]),self.alpha)
                np.testing.assert_array_equal(before,after)

    def test_current_future_economic_components_cannot_change_structural_origin(self):
        for t in (56,57,60,61,67):
            expected=m.component_structural_prior(
                self.num,self.den,self.info,np.array([t]),self.alpha)
            num=self.num.copy()
            den=self.den.copy()
            num[t:]+=1000.
            den[t:,:,:2]*=4.
            actual=m.component_structural_prior(
                num,den,self.info,np.array([t]),self.alpha)
            np.testing.assert_array_equal(expected,actual)

    def test_future_annual_gos_is_not_used_to_reconstruct_prior_profitability(self):
        original_reader=pd.read_csv

        def reader(path,*args,**kwargs):
            result=original_reader(path,*args,**kwargs)
            if Path(path)!=m.STAGE04_COMPONENT_SOURCE:
                return result
            def chunks():
                for block in result:
                    block=block.copy()
                    future=(block.provider.eq("BEA") & block.series.eq("gos") &
                            block.frequency.eq("annual") & block.period.eq("2022"))
                    block.loc[future,"value"]*=20.
                    yield block
            return chunks()

        with patch.object(m.pd,"read_csv",side_effect=reader):
            num,den,info=m.economic_components(self.panel)
        np.testing.assert_array_equal(num,self.num)
        np.testing.assert_array_equal(den,self.den)
        for key in info:
            if isinstance(info[key],np.ndarray):
                np.testing.assert_array_equal(info[key],self.info[key])
            else:
                self.assertEqual(info[key],self.info[key])

    def test_train_graph_ignores_retrospective_target_and_dev_predictors(self):
        original=m.train_graph(self.panel)
        p=self.panel
        changed=m.Panel(p.x.copy(),p.y.copy(),p.quarters,
                        p.nodes,p.names,p.features,p.train_count,p.hashes)
        changed.y+=100.
        changed.x[56:,:,3:]+=1000.
        np.testing.assert_array_equal(original,m.train_graph(changed))

    def test_alpha_selection_uses_full_train_not_an_inner_2017_cut(self):
        p=self.panel
        expected,table=m.select_train_structural_priors(
            p,self.num,self.den,self.info,fit_end=56)
        self.assertTrue(table.alpha_decision_quarters.eq(56).all())
        self.assertTrue(table.alpha_decision_years.eq(14).all())
        with self.assertRaisesRegex(ValueError,"ALL 2006-2019 Train"):
            m.select_train_structural_priors(
                p,self.num,self.den,self.info,fit_end=48)
        modified=m.Panel(p.x.copy(),p.y.copy(),p.quarters,
                         p.nodes,p.names,p.features,p.train_count,p.hashes)
        modified.y[56:]+=9999.  # Development labels must not select alpha.
        actual,other=m.select_train_structural_priors(
            modified,self.num,self.den,self.info,fit_end=56)
        np.testing.assert_array_equal(expected,actual)
        pd.testing.assert_frame_equal(table,other,check_exact=True)

    def test_scaling_excludes_dev_and_test(self):
        p=self.panel
        center,width=m.fit_normalizer(p.x[:56],"standard")
        altered=p.x.copy()
        altered[56:]+=1000.
        c,w=m.fit_normalizer(altered[:56],"standard")
        np.testing.assert_array_equal(center,c)
        np.testing.assert_array_equal(width,w)


if __name__=="__main__":
    unittest.main()
