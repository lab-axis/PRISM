"""PRISM06 full-factorial 216-grid scientific contract; no legacy mode."""
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
    "_prism06_factorial_216_tests",ROOT/"Sources/06_preprocessing.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def fabricated_scores():
    """All 216 cells synthetic: verifies search and plots without GPU fits."""
    policies=m.enumerate_policies()
    synergy=m.Policy("train_mean","robust_mad","asinh")
    rows={}
    for policy in policies:
        i,j,k=(m.MISSING.index(policy.missing),
               m.NORMALIZATION.index(policy.normalization),
               m.DISTRIBUTION.index(policy.distribution))
        mae=2.07 + .015*i + .013*j + .008*k
        ici=1.86 + .004*i + .003*j + .002*k
        if policy==m.ORIGINAL:
            mae=2.0
            ici=1.80
        if policy==synergy:
            mae=1.8
            ici=1.70
        result=dict(
            policy=policy.id,**asdict_for_test(policy),
            status="PASS",
            axis_mae=mae,ici_mae=ici,
            axis_mae_seed_sd=.004,ici_mae_seed_sd=.003,
            profitability_mae=mae*.95,
            productivity_mae=mae*.65,
            growth_mae=mae*1.4,
            **{f"{year}_mae":mae+.01*ii
               for ii,year in enumerate((2020,2021,2022))},
            **{f"{year}_ici_mae":ici+.01*ii
               for ii,year in enumerate((2020,2021,2022))},
            seed_results=[
                dict(axis_mae=mae+(seed-2030.5)*.001,
                     ici_mae=ici+(seed-2030.5)*.001)
                for seed in m.SEEDS],
            input_sha256=policy.id,
            identical_input_reused_from=None)
        rows[policy.id]=result
    return policies,rows,synergy


def asdict_for_test(policy):
    return {key:getattr(policy,key) for key in m.SLOTS}


class FactorialDesignTests(unittest.TestCase):
    def test_216_cartesian_no_sequential_or_legacy(self):
        grid=m.enumerate_policies()
        self.assertEqual(m.GRID_SIZE,216)
        self.assertEqual(len(grid),216)
        self.assertEqual(len({p.id for p in grid}),216)
        self.assertEqual(grid[0],m.ORIGINAL)
        self.assertEqual(
            {p.missing for p in grid},set(m.MISSING))
        self.assertEqual(
            {p.normalization for p in grid},set(m.NORMALIZATION))
        self.assertEqual(
            {p.distribution for p in grid},set(m.DISTRIBUTION))
        self.assertEqual(len(m.SEEDS),10)
        self.assertFalse(hasattr(m,"LOOKBACK"))
        self.assertEqual(m.SLOTS,("missing","normalization","distribution"))
        self.assertFalse(hasattr(m,"greedy_next"))
        self.assertFalse(hasattr(m,"factor_candidates"))
        self.assertFalse(hasattr(m,"ablated_policies"))
        self.assertFalse(hasattr(m,"OUTLIER"))
        self.assertFalse(hasattr(m.Policy(),"outlier"))
        self.assertFalse(hasattr(m,"verify_04_and_load"))

    def test_full_grid_finds_interaction_benefit_missed_by_greedy(self):
        policies,rows,synergy=fabricated_scores()
        table=m.eligible_policy_table(policies,rows)
        self.assertEqual(len(table),216)
        self.assertTrue(table.loc[
            table.policy.eq(synergy.id),"eligible"].iloc[0])
        self.assertEqual(m.select_grid_policy(table),synergy)
        self.assertEqual(int(table.eligible.sum()),2)
        # Every single-change neighbor of the baseline is worse.
        single=[p for p in policies if sum(
            getattr(p,f)!=getattr(m.ORIGINAL,f) for f in m.SLOTS)==1]
        self.assertTrue(all(
            not table.loc[table.policy.eq(p.id),"eligible"].iloc[0]
            for p in single))

    def test_global_grid_eligibility_is_against_original_not_incumbent(self):
        policies,rows,synergy=fabricated_scores()
        another=m.Policy("train_median","robust_iqr","mad_clip5")
        row=rows[another.id]
        row.update(axis_mae=1.7,ici_mae=1.81,
                   **{f"{y}_mae":1.7 for y in (2020,2021,2022)})
        table=m.eligible_policy_table(policies,rows)
        self.assertFalse(table.set_index("policy").loc[another.id,"eligible"])
        row["ici_mae"]=1.70
        row["2020_mae"]=2.01
        row["2021_mae"]=2.02
        row["2022_mae"]=1.7
        table=m.eligible_policy_table(policies,rows)
        self.assertFalse(table.set_index("policy").loc[another.id,"eligible"])
        row["2021_mae"]=1.7
        table=m.eligible_policy_table(policies,rows)
        self.assertTrue(table.set_index("policy").loc[another.id,"eligible"])
        self.assertEqual(m.select_grid_policy(table),another)

    def test_incomplete_grid_fails_closed(self):
        policies,rows,_=fabricated_scores()
        with self.assertRaises(RuntimeError):
            m.eligible_policy_table(policies[:-1],rows)
        with self.assertRaises(RuntimeError):
            m.eligible_policy_table(policies,{
                p.id:rows[p.id] for p in policies[:-1]})
        failed=policies[-1]
        rows[failed.id]=dict(
            policy=failed.id,status="NUMERICAL_FAILURE",
            input_sha256="test-failure",error="nonfinite input")
        table=m.eligible_policy_table(policies,rows)
        self.assertFalse(table.iloc[-1].eligible)
        self.assertTrue(np.isnan(table.iloc[-1].axis_mae))
        self.assertNotEqual(m.select_grid_policy(table),failed)

    def test_pairwise_interaction_residuals_and_pareto(self):
        policies,rows,synergy=fabricated_scores()
        table=m.eligible_policy_table(policies,rows)
        interactions=m.pairwise_interaction_tables(table)
        self.assertEqual(len(interactions),3*6*6)
        self.assertTrue(interactions.grid_complete_at_pair.all())
        self.assertGreater(
            np.abs(interactions.two_factor_residual_mae).max(),0.001)
        pareto=m.descriptive_pareto(table)
        self.assertIn(synergy.id,set(pareto.policy))
        self.assertEqual(pareto.iloc[0].policy,synergy.id)

    def test_all_descriptive_plots_from_full_216_synthetic(self):
        policies,rows,chosen=fabricated_scores()
        table=m.eligible_policy_table(policies,rows)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            images=m.chart_full_factorial(
                target,table,rows,chosen,None)
            self.assertEqual(len(images),11)
            self.assertEqual(len(set(images)),11)
            self.assertTrue(images[0].endswith("registered_216_grid.png"))
            for label in images:
                file=target/label
                self.assertTrue(file.is_file(),label)
                self.assertGreater(file.stat().st_size,10000)
            self.assertEqual(
                len(pd.read_csv(target/"factor_interactions.csv")),108)
            self.assertEqual(
                len(pd.read_csv(target/"per_seed_policy_scores.csv")),2160)
            self.assertGreaterEqual(
                len(pd.read_csv(target/"mae_ici_pareto.csv")),1)


class RealPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_current_results(5)
        torch.set_num_threads(3)
        cls.m4,cls.m5,cls.panel,cls.lock,cls.manifest=(
            m.verify_upstream_and_load())
        cls.prior=m.structural_prior(cls.panel)

    def test_exact_original_stage05_inputs_and_predictions(self):
        self.assertTrue(m.original_stage05_parity(self.panel,self.prior))
        standard,_=m.prepare_policy(self.panel,m.ORIGINAL,self.prior)
        length=m.selected_lookback()
        canonical=self.m5.Inputs(self.panel).fold(length,full_train=True)
        for key in canonical:
            if key in standard:
                np.testing.assert_array_equal(
                    standard[key],canonical[key])
        self.assertEqual(standard["train_x"].shape,(len(m.train_origins()),length,59,157))
        self.assertEqual(standard["val_x"].shape,(12,length,59,157))
        preds=m.original_stage05_seed_predictions(self.panel)
        score,indiv=m.seed_policy_metrics(preds,self.panel)
        self.assertEqual(preds.shape,(10,12,59,3))
        self.assertEqual(len(indiv),10)
        self.assertAlmostEqual(score["axis_mae"],np.abs(preds-self.panel.y[56:68]).mean(),9)

    def test_distribution_transform_affects_only_source_columns(self):
        raw=self.panel.x
        baseline=np.nan_to_num(raw[:,:,:42])
        for kind in m.DISTRIBUTION:
            altered=m.distribution_transform(
                raw,m.Policy(distribution=kind),m.TRAIN_END)
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(
                    np.nan_to_num(altered[:,:,:42]),baseline)

    def test_train_fitted_only_and_no_future_leakage(self):
        corrupted=self.panel.x.copy()
        corrupted[56:,:,42:]=np.where(
            np.isfinite(corrupted[56:,:,42:]),
            corrupted[56:,:,42:]+1e8,np.nan)
        modified=replace(self.panel,x=corrupted)
        for p in (
            m.ORIGINAL,
            m.Policy("train_mean","quantile_normal","signed_log1p"),
            m.Policy("causal_seasonal","robust_iqr","winsor01"),
            m.Policy("scaled_zero","robust_mad","mad_clip5")):
            with self.subTest(policy=p.id):
                first,_=m.prepare_policy(self.panel,p,self.prior)
                future,_=m.prepare_policy(modified,p,self.prior)
                for key in ("train_x","train_mask","train_y",
                            "train_context","train_last","train_structure",
                            "x_center","x_scale","y_center","y_scale"):
                    np.testing.assert_array_equal(first[key],future[key])
                np.testing.assert_array_equal(
                    first["val_x"][0],future["val_x"][0])
        with self.assertRaises(ValueError):
            m.prepare_policy(self.panel,m.ORIGINAL,self.prior,68)

    def test_source_prefix_duplicates_and_policy_hash(self):
        train_median=m.Policy("train_median","standard","none")
        forward=m.Policy("causal_ffill","standard","none")
        seasonal=m.Policy("causal_seasonal","standard","none")
        expected=m.input_fingerprint(
            m.prepare_policy(self.panel,train_median,self.prior)[0])
        self.assertEqual(
            m.input_fingerprint(
                m.prepare_policy(self.panel,forward,self.prior)[0]),
            expected)
        self.assertEqual(
            m.input_fingerprint(
                m.prepare_policy(self.panel,seasonal,self.prior)[0]),
            expected)

    def test_registered_grid_uses_current_L1_and_train_only_provenance(self):
        contract,digest=m.configuration(
            self.m4,self.m5,self.lock,self.manifest,self.panel)
        self.assertEqual(contract["experiment"],
                         "PRISM06_FULL_FACTORIAL_216_SINGLE_CANONICAL")
        self.assertEqual(contract["cartesian_policy_count"],216)
        self.assertEqual(contract["lookback"],m.selected_lookback())
        self.assertEqual(contract["seed_grid"],list(m.SEEDS))
        self.assertEqual(contract["fixed_epochs"],75)
        self.assertEqual(contract["train_origin_indices"],
                         m.train_origins().tolist())
        self.assertEqual(contract["factor_names"],list(m.SLOTS))
        self.assertFalse(contract["test_opened"])
        self.assertTrue(contract["no_train_cv"])
        self.assertEqual(len(digest),64)
        self.assertNotIn("raw_test.csv",m.SOURCE.read_text(encoding="utf8"))

    def test_single_seed_training_smoke_for_unseen_combination(self):
        altered=m.Policy("scaled_zero","robust_iqr","asinh")
        arrays,_=m.prepare_policy(self.panel,altered,self.prior)
        with patch.object(self.m4,"STAGE04_ACTIVE_EPOCHS",1):
            out,state,seconds=m.fit_policy_model(
                arrays,2026,torch.device("cpu"))
        self.assertEqual(out.shape,(12,59,3))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(state)
        self.assertGreater(seconds,0)

    def test_result_folder_only_accepts_exact_current_contract(self):
        require_current_results(6)
        # An unfinished 216-way run is valid; old sequential status is not.
        marker=m.RESULTS/"run_manifest.json"
        if not marker.is_file():
            self.skipTest("No full-factorial results have been started")
        run=m.json_read(marker)
        self.assertEqual(run["schema"],"PRISM06_FULL_FACTORIAL_216")
        self.assertEqual(
            run["contract"]["experiment"],
            "PRISM06_FULL_FACTORIAL_216_SINGLE_CANONICAL")
        self.assertLessEqual(run["candidates_complete"],216)
        self.assertFalse(run["test_opened"])


if __name__=="__main__":
    unittest.main()
