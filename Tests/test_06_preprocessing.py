"""PRISM 06 sequential preprocessing + Train-only distribution leakage audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_prism06_preprocessing_tests",ROOT/"Sources/06_preprocessing.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class DesignTests(unittest.TestCase):
    def test_exactly_five_alternatives_plus_current(self):
        self.assertEqual(len(m.MISSING),6)
        self.assertEqual(len(m.NORMALIZATION),6)
        self.assertEqual(len(m.OUTLIER),6)
        base=m.ORIGINAL
        self.assertEqual(base,m.Policy("raw_zero","standard","none"))
        for stage in (1,2,3):
            candidates=m.factor_candidates(stage,base)
            self.assertEqual(len(candidates),6)
            self.assertEqual(candidates[0],base)
            self.assertEqual(len(set(p.id for p in candidates)),6)
            for p in candidates[1:]:
                self.assertEqual(
                    sum(getattr(p,k)!=getattr(base,k) for k in m.SLOTS),1)
        self.assertEqual(
            len(m.factor_candidates(2,m.Policy(missing="scaled_zero"))),6)

    def test_conservative_step_uses_years_and_ici(self):
        def point(mae=2.,ici=1.5,year=(2.,2.,2.),status="PASS"):
            return dict(status=status,axis_mae=mae,ici_mae=ici,
                        **{f"{y}_mae":v for y,v in zip(
                            (2020,2021,2022),year)})
        c=m.factor_candidates(1,m.ORIGINAL)
        data={c[0].id:point()}
        data[c[1].id]=point(1.7,1.4,(1.7,1.8,2.2))
        data[c[2].id]=point(1.4,1.6,(1.3,1.5,1.4)) # ICI worsens
        data[c[3].id]=point(1.3,1.3,(1.5,2.2,2.1)) # only one year
        data[c[4].id]=point(2.1,1.4,(1.9,1.9,2.4))
        data[c[5].id]=dict(status="NUMERICAL_FAILURE")
        winner,rows=m.greedy_next(m.ORIGINAL,1,data)
        self.assertEqual(winner,c[1])
        self.assertEqual(len(rows),6)
        self.assertEqual(sum(row["stage_result"] for row in rows),1)
        self.assertEqual(rows[-1]["status"],"NUMERICAL_FAILURE")
        self.assertIsNone(rows[-1]["axis_mae"])
        data[c[1].id]=point(1.7,1.6,(1.7,1.8,2.2))
        chosen,_=m.greedy_next(m.ORIGINAL,1,data)
        self.assertEqual(chosen,m.ORIGINAL)

    def test_reverse_ablation_returns_only_changed_components(self):
        self.assertEqual(m.ablated_policies(m.ORIGINAL),())
        p=m.Policy("scaled_zero","robust_mad","winsor01")
        variants=m.ablated_policies(p)
        self.assertEqual(len(variants),3)
        for r in variants:
            self.assertEqual(
                sum(getattr(r,k)!=getattr(p,k) for k in m.SLOTS),1)
            self.assertNotEqual(r,p)
        self.assertEqual(variants[0].missing,m.ORIGINAL.missing)


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.original,cls.panel,cls.models,cls.epochs,_,_=(
            m.verify_04_and_load())
        cls.prior=m.structural_prior(cls.panel)

    def test_exact_stage04_baseline_preprocessing(self):
        self.assertTrue(m.original_04_parity(self.panel,self.prior))
        old=m.original_stage04_prediction(self.panel,self.models)
        scores=m.scores(
            old,self.panel.y[56:68],
            self.original.past_scores(self.panel,np.arange(56,68)))
        self.assertAlmostEqual(scores["axis_mae"],2.340279421959421,places=9)
        self.assertAlmostEqual(scores["ici_mae"],1.7894925355233664,places=9)

    def test_all_six_missing_and_scaling_candidates_are_finite(self):
        for variant in m.MISSING:
            policy=m.Policy(missing=variant)
            with self.subTest(variant=variant):
                data,audit=m.prepare_policy(self.panel,policy,self.prior)
                self.assertEqual(data["train_x"].shape,(44,12,59,157))
                self.assertEqual(data["val_x"].shape,(12,12,59,157))
                self.assertTrue(np.isfinite(data["train_x"]).all())
                self.assertTrue(np.isfinite(data["val_x"]).all())
                self.assertTrue(audit["original_missingness_preserved"])
        for variant in m.NORMALIZATION:
            policy=m.Policy(normalization=variant)
            with self.subTest(variant=variant):
                data,_=m.prepare_policy(self.panel,policy,self.prior)
                self.assertTrue(np.isfinite(data["train_x"]).all())
                self.assertTrue(np.isfinite(data["val_x"]).all())
        for variant in m.OUTLIER:
            policy=m.Policy(outlier=variant)
            with self.subTest(variant=variant):
                data,_=m.prepare_policy(self.panel,policy,self.prior)
                self.assertTrue(np.isfinite(data["train_x"]).all())

    def test_per_feature_scaler_is_train_only_even_when_dev_distribution_poisoned(self):
        p=self.panel
        future_x=p.x.copy()
        future_x[56:,:,42:]=np.where(
            np.isfinite(future_x[56:,:,42:]),
            future_x[56:,:,42:]+1e8,np.nan)
        future=self.original.Panel(
            future_x,p.y.copy(),p.quarters,p.nodes,p.names,
            p.features,p.train_count,p.hashes)
        for variant in m.NORMALIZATION:
            policy=m.Policy(missing="scaled_zero",normalization=variant)
            with self.subTest(scaling=variant):
                original,_=m.prepare_policy(p,policy,self.prior)
                poisoned,_=m.prepare_policy(future,policy,self.prior)
                for key in (
                        "train_x","train_mask","train_y",
                        "x_center","x_scale","y_center","y_scale"):
                    np.testing.assert_array_equal(original[key],poisoned[key])
                # 2020Q1 inputs must be the SAME even if 2020-22 X changes.
                np.testing.assert_array_equal(
                    original["val_x"][0],poisoned["val_x"][0])

    def test_future_values_do_not_backfill_earlier_val_quarters(self):
        p=self.panel
        altered_x=p.x.copy()
        altered_x[60:,:,42:]=np.where(
            np.isfinite(altered_x[60:,:,42:]),
            altered_x[60:,:,42:]*100.+200,np.nan)
        variant=self.original.Panel(
            altered_x,p.y,p.quarters,p.nodes,p.names,p.features,
            p.train_count,p.hashes)
        for how in ("raw_zero","scaled_zero","train_median",
                    "train_mean","causal_ffill","causal_seasonal"):
            pol=m.Policy(missing=how)
            original,_=m.prepare_policy(p,pol,self.prior)
            poison,_=m.prepare_policy(variant,pol,self.prior)
            with self.subTest(missing=how):
                np.testing.assert_array_equal(
                    original["train_x"],poison["train_x"])
                # Forecast origins 2020Q1..2021Q1 never include X[2021Q1].
                np.testing.assert_array_equal(
                    original["val_x"][:5],poison["val_x"][:5])
                self.assertFalse(
                    np.array_equal(original["val_x"][6:],poison["val_x"][6:]))

    def test_outliers_do_not_modify_42_target_derived_history_channels(self):
        observed=np.isfinite(self.panel.x)
        for kind in m.OUTLIER:
            if kind=="none":continue
            p=m.Policy(outlier=kind)
            transformed=m.outlier_transform(self.panel.x,p,m.TRAIN_END)
            with self.subTest(outlier=kind):
                np.testing.assert_array_equal(
                    np.where(observed[:,:,:42],
                             transformed[:,:,:42],0.),
                    np.where(observed[:,:,:42],
                             self.panel.x[:,:,:42],0.))
                self.assertTrue(np.array_equal(
                    np.isnan(transformed[:,:,:42]),
                    np.isnan(self.panel.x[:,:,:42])))

    def test_causal_imputer_does_not_fill_derived_labels(self):
        obs=np.isfinite(self.panel.x)
        for name in m.MISSING:
            adjusted=m.causal_source_impute(
                self.panel.x,obs,m.Policy(missing=name),56)
            with self.subTest(missing=name):
                np.testing.assert_array_equal(
                    np.isnan(adjusted[:,:,:42]),
                    np.isnan(self.panel.x[:,:,:42]))

    def test_validation_distribution_cannot_be_fitted_in_safety_gate(self):
        with self.assertRaises(ValueError):
            m.observed_train_statistics(self.panel.x,68)
        with self.assertRaises(ValueError):
            m.outlier_transform(self.panel.x,m.Policy(outlier="winsor01"),68)
        with self.assertRaises(ValueError):
            m.prepare_policy(self.panel,m.ORIGINAL,self.prior,68)

    def test_source_missingness_is_only_initial_prefix_no_intervening_gap(self):
        source=np.isfinite(self.panel.x[:,:,42:])
        seen=np.zeros(source.shape[1:],dtype=bool)
        missing_after_observed=0
        for t in range(len(source)):
            missing_after_observed+=int((~source[t]&seen).sum())
            seen|=source[t]
        self.assertEqual(missing_after_observed,0)
        # In THIS data, true causal ffill/seasonal have no recorded past
        # source value at any missing cell and therefore reduce to the
        # frozen Train median fallback. Do not claim separate improvements.
        baseline,_=m.prepare_policy(
            self.panel,m.Policy(missing="train_median"),self.prior)
        for name in ("causal_ffill","causal_seasonal"):
            candidate,_=m.prepare_policy(
                self.panel,m.Policy(missing=name),self.prior)
            np.testing.assert_array_equal(
                baseline["train_x"],candidate["train_x"])
            np.testing.assert_array_equal(
                baseline["val_x"],candidate["val_x"])


class ProductionResultTests(unittest.TestCase):
    def test_full_stage06_completed_only_train_fitted_frozen(self):
        output=m.RESULTS
        manifest=m.json_read(output/"run_manifest.json")
        selection=m.json_read(output/"selected_preprocessing.json")
        self.assertEqual(manifest["status"],"PASS")
        self.assertEqual(selection["status"],"LOCKED")
        self.assertEqual(manifest["selected_policy"],m.ORIGINAL.id)
        self.assertEqual(selection["policy_id"],m.ORIGINAL.id)
        self.assertTrue(selection["no_fitted_distributions_from_dev"])
        self.assertTrue(selection["no_train_internal_evaluation"])
        self.assertTrue(selection["frozen_weights_for_every_dev_quarter"])
        self.assertFalse(selection["test_opened"])
        self.assertEqual(selection["total_policy_lookups"],16)
        self.assertEqual(len(list((output/"jobs").rglob("*.pt"))),375)
        self.assertEqual(len(list((output/"jobs").rglob("*.npz"))),375)
        self.assertEqual(len(list((output/"jobs").rglob("*.json"))),375)
        self.assertEqual(len(list((output/"plots").glob("*.png"))),5)

    def test_all_three_stage_tables_have_six_candidates_and_no_leak(self):
        for i,axis in enumerate(("missing","normalization","outlier"),1):
            table=pd.read_csv(
                m.RESULTS/f"stage{i:02d}_{axis}_comparison.csv")
            self.assertEqual(len(table),6)
            self.assertEqual(len(set(table.policy)),6)
            self.assertEqual(table.stage_result.sum(),1)
            self.assertEqual(table.qualifies_as_change.sum(),0)
            self.assertEqual(
                table.loc[table.stage_result,"policy"].iloc[0],m.ORIGINAL.id)
            decision=m.json_read(m.RESULTS/f"stage{i:02d}_decision.json")
            self.assertEqual(decision["retained_policy"],m.ORIGINAL.id)
            self.assertTrue(decision["dev_selection_not_independent_test"])

    def test_selected_predictions_reuse_stage04_exactly(self):
        panel=m.stage04().load_panel(ROOT)
        name=m.RESULTS/"frozen_selected_vs_04_dev.npz"
        with np.load(name,allow_pickle=False) as z:
            # 06 averages seeds first, then GNNs; the published 04 cache
            # averages GNNs first, then seeds. Both are mathematically
            # equal; their floating-point order may differ at ~1e-14.
            np.testing.assert_allclose(
                z["prediction"],z["original_04_prediction"],
                atol=1e-12,rtol=0)
            np.testing.assert_array_equal(
                z["truth"],panel.y[56:68])

    def test_stage04_locked_epochs_and_hashes_remain_frozen(self):
        _,panel,models,epochs,lock,manifest=m.verify_04_and_load()
        report=m.json_read(m.RESULTS/"run_manifest.json")
        self.assertEqual(report["contract"]["stage04_source_sha256"],
                         m.sha(m.SOURCE04))
        for model in models:
            for seed in m.SEEDS:
                key=f"{model}__s{seed}"
                self.assertEqual(
                    report["contract"]["fixed_epochs"][key],
                    epochs[(model,seed)])
        self.assertEqual(report["contract"]["input_quarters"],12)
        self.assertEqual(report["contract"]["target_horizon_quarters"],1)


if __name__=="__main__":
    unittest.main()
