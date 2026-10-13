"""PRISM Stage05: matched-capacity B-MTGNN lookback sensitivity, Dev only."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_prism05_bmtgnn_tests",ROOT/"Sources/05_input_lookback.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class DesignTests(unittest.TestCase):
    def test_single_model_ten_seeds_and_predeclared_common_support(self):
        self.assertEqual(m.RESULTS,ROOT/"Results/05_input_lookback")
        self.assertEqual(m.MODEL,"B-MTGNN")
        self.assertEqual(m.MODE,"structural")
        self.assertEqual(m.SOURCE04,ROOT/"Sources/04_bmtgnn_benchmark.py")
        self.assertEqual(m.SEEDS,tuple(range(2026,2036)))
        self.assertEqual(m.LENGTHS,tuple(range(1,21)))
        self.assertEqual(m.ORIGINAL_LOOKBACK,12)
        self.assertEqual(m.CORE_LOOKBACK,20)
        self.assertEqual(m.COMMON_TRAIN_ORIGINS.tolist(),list(range(20,56)))
        self.assertEqual(m.DEV_ORIGINS.tolist(),list(range(56,68)))
        design=m.study_design()
        self.assertEqual(design["epoch_selection"],"NONE_FIXED_75")
        self.assertFalse(design["inner_validation"])
        self.assertEqual(design["train_heldout_years"],[])
        self.assertTrue(design["same_model_parameters_all_lengths"])
        self.assertTrue(design["common_asof_9d_target_context"])
        self.assertTrue(design["dev_used_for_main_policy_selection"])
        self.assertFalse(design["train_internal_cv"])
        self.assertEqual(design["training_strategy"],"SINGLE_FULL_TRAIN_FIT")
        self.assertFalse(hasattr(m,"TRAIN_SELECTION_FOLDS"))
        self.assertFalse(hasattr(m,"TRAIN_SELECTION_ORIGINS"))
        self.assertEqual(
            design["experiment"],
            "conditional_quarterly_feature_window_not_total_history")

    def test_current_stage04_dev_freeze_and_ten_seed_lock(self):
        m4,panel,selection,manifest=m.upstream()
        self.assertEqual(m4.GNN_MODELS,("B-MTGNN",))
        self.assertEqual(selection["selected_mode"],m.MODE)
        self.assertEqual(selection["seed_count"],10)
        self.assertEqual(manifest["status"],"DEV_FROZEN")
        self.assertEqual(manifest["completed_jobs"],30)
        self.assertEqual(manifest["total_jobs"],30)
        self.assertEqual(panel.x.shape,(68,59,157))
        self.assertNotIn(
            "Data/03_integrated_dataset/raw_test.csv",panel.hashes)
        contract,digest=m.run_contract(m4,panel,selection,manifest)
        self.assertEqual(contract["stage04_trained_source_sha256"],
                         manifest["training_source_sha256"])
        self.assertEqual(contract["fixed_model_shape_every_length"],True)
        self.assertEqual(contract["seeds"],list(m.SEEDS))
        self.assertEqual(contract["fixed_epochs"],75)
        self.assertEqual(len(digest),64)


class InputsAndInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.m4,cls.panel,_,_=m.upstream()
        cls.inputs=m.Inputs(cls.panel)

    def test_all_lengths_have_identical_train_targets_and_dev_origins(self):
        a=self.inputs.fold(12)
        for length in m.LENGTHS:
            b=self.inputs.fold(length)
            with self.subTest(L=length):
                self.assertEqual(b["train_x"].shape,(36,length,59,157))
                self.assertEqual(b["val_x"].shape,(12,length,59,157))
                self.assertEqual(b["train_y"].shape,(36,1,59,3))
                self.assertEqual(b["val_y"].shape,(12,1,59,3))
                self.assertEqual(b["train_origins"].tolist(),list(range(20,56)))
                self.assertEqual(b["val_origins"].tolist(),list(range(56,68)))
                for key in ("train_y","train_last","train_context",
                            "train_structure","val_y","val_last",
                            "val_context","val_structure","x_center",
                            "x_scale","y_center","y_scale"):
                    np.testing.assert_array_equal(a[key],b[key])
                self.assertTrue(np.isfinite(b["train_x"]).all())
                self.assertTrue(np.isfinite(b["val_x"]).all())
        self.assertEqual(a["train_origins"][-8:].tolist(),list(range(48,56)))
        np.testing.assert_array_equal(
            a["train_x"][:,-1],self.inputs.fold(1)["train_x"][:,0])

    def test_stage04_original_dev_inputs_and_economic_prior_match(self):
        panel=self.panel
        economic,_=self.m4.make_fold(panel)
        np.testing.assert_array_equal(self.inputs.prior,economic)
        original=self.m4.prepare_three_axis_fold(
            panel,economic,56,56,68,"standard","mask")
        stage05=self.inputs.fold(12)
        for key in ("x_center","x_scale","y_center","y_scale",
                    "val_x","val_mask","val_y","val_last",
                    "val_context","val_structure"):
            with self.subTest(field=key):
                np.testing.assert_array_equal(stage05[key],original[key])
        self.assertNotEqual(original["train_x"].shape[0],
                            stage05["train_x"].shape[0])

    def test_no_train_cv_or_shortened_train_fits_can_be_invoked(self):
        self.assertFalse(hasattr(self.inputs,"train_prefix_prior"))
        with self.assertRaises(TypeError):
            self.inputs.fold(12,fit_end=40,eval_end=48)
        z=self.inputs.fold(12)
        self.assertEqual(z["train_origins"][0],20)
        self.assertEqual(z["train_origins"][-1],55)
        self.assertEqual(z["val_origins"][0],56)
        self.assertEqual(z["val_origins"][-1],67)

    def test_poisoned_dev_labels_and_features_do_not_change_train_fit(self):
        before=self.inputs.fold(12)
        p=self.panel
        corrupted_x=p.x.copy()
        corrupted_y=p.y.copy()
        corrupted_x[56:,:,42:]+=1e7
        corrupted_y[56:]+=1e6
        poisoned=replace(p,x=corrupted_x,y=corrupted_y)
        self.inputs.panel=poisoned
        try:
            after=self.inputs.fold(12)
        finally:
            self.inputs.panel=p
        for key in ("train_x","train_mask","train_y","train_last",
                    "train_context","train_structure","x_center",
                    "x_scale","y_center","y_scale","val_last",
                    "val_context","val_structure"):
            np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_array_equal(before["val_x"][0],after["val_x"][0])
        self.assertFalse(np.array_equal(before["val_y"],after["val_y"]))

    def test_every_length_has_exactly_same_initial_parameter_tensors(self):
        baseline=None
        baseline_rng=None
        for L in m.LENGTHS:
            with self.subTest(L=L):
                data=self.inputs.fold(L)
                net=m.instantiate(self.m4,m.MODEL,2026,L,data,m.MODE)
                state=net.state_dict()
                rng=torch.get_rng_state().clone()
                if baseline is None:
                    baseline={k:v.clone() for k,v in state.items()}
                    baseline_rng=rng
                self.assertEqual(set(state),set(baseline))
                for key,value in state.items():
                    torch.testing.assert_close(
                        value,baseline[key],atol=0,rtol=0)
                self.assertTrue(torch.equal(rng,baseline_rng))
                self.assertEqual(
                    sum(x.numel() for x in net.parameters()),67190)
                self.assertEqual(
                    net.base.core.backbone.skip0.kernel_size[1],20)
        self.assertEqual(m.ORIGINAL_LOOKBACK,12)

    def test_padding_is_zero_latent_states_not_fake_observed_rows(self):
        class Capture(torch.nn.Module):
            def forward(self,encoded,graph):
                self.last=encoded.clone()
                return encoded
        monitor=Capture()
        wrapper=m.FixedTemporalInput(monitor)
        for L in (1,12,19,20):
            with self.subTest(L=L):
                valid=torch.randn(2,L,59,16)
                output=wrapper(valid,torch.eye(59))
                self.assertEqual(output.shape,(2,20,59,16))
                torch.testing.assert_close(
                    output[:,20-L:],valid,atol=0,rtol=0)
                self.assertEqual(
                    torch.count_nonzero(output[:,:20-L]).item(),0)

    def test_the_model_really_consumes_one_to_twenty_quarters(self):
        for length in (1,12,19,20):
            with self.subTest(L=length):
                data=self.inputs.fold(length)
                net=m.instantiate(
                    self.m4,m.MODEL,2026,length,data,m.MODE)
                net.eval()
                z={k:torch.as_tensor(data["val_"+k][:1])
                   for k in ("x","mask","last","structure","context")}
                with torch.no_grad():
                    y=net(z["x"],z["mask"],torch.eye(59),
                          z["last"],z["structure"],z["context"])
                self.assertEqual(y.shape,(1,1,59,3))
                self.assertTrue(torch.isfinite(y).all())

    def test_one_epoch_training_smoke_has_valid_mc_prediction(self):
        with patch.object(self.m4,"STAGE04_ACTIVE_EPOCHS",1):
            predicted=m.train_one(
                self.inputs,m.MODEL,2026,20,torch.device("cpu"))
        self.assertEqual(predicted.shape,(12,59,3))
        self.assertTrue(np.isfinite(predicted).all())


class SeedAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=m.stage04().load_panel(ROOT)

    def test_errors_are_scored_before_seed_aggregation_and_paired(self):
        rows=[]
        for L in m.LENGTHS:
            for seed in m.SEEDS:
                bias=(seed-2026-4.5)*.1+L*.001
                v=abs(bias)
                row=dict(input_quarters=L,seed=seed,model=m.MODEL,
                         mode=m.MODE,axis_mae=v,ici_mae=v,
                         **{f"{axis}_mae":v for axis in m.SCORE_NAMES},
                         **{f"{year}_{metric}":v
                            for year in (2020,2021,2022)
                            for metric in ("mae","ici_mae")})
                rows.append(row)
        scored=pd.DataFrame(rows)
        result=m.summarize_seed_scores(scored)
        self.assertEqual(len(result),20)
        self.assertTrue((result.seed_count==10).all())
        self.assertGreater(result.iloc[11].axis_mae,.20)
        # Seed-ensemble prediction errors would be ~.012, incorrectly small.
        self.assertNotAlmostEqual(result.iloc[11].axis_mae,.012,places=2)
        self.assertAlmostEqual(
            result.loc[result.input_quarters==12,
                       "axis_mae_paired_delta_vs_12q"].iloc[0],0.)
        self.assertAlmostEqual(
            result.loc[result.input_quarters==12,
                       "axis_mae_paired_delta_sd"].iloc[0],0.)

    def test_200_job_dev_selection_and_four_descriptive_plots(self):
        m4=m.stage04()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            digest="unit_test_05_signature"
            truth=self.panel.y[56:68]
            for L in m.LENGTHS:
                for seed in m.SEEDS:
                    stem=f"L{L:02d}__{m.MODEL}__s{seed}"
                    path=root/"trials"/(stem+".npz")
                    prediction=truth+(.02*L+.1*(seed-2030))
                    m.save_npz(path,prediction=prediction)
                    m.json_write(root/"trials"/(stem+".json"),dict(
                        signature=digest,L=L,model=m.MODEL,seed=seed,
                        fixed_epochs=75,prediction_sha256=m.sha(path),
                        train_internal_validation=False))
            seed_scores=m.score_results(root,digest,self.panel)
            self.assertEqual(len(seed_scores),200)
            row=seed_scores[
                (seed_scores.input_quarters==12)&
                (seed_scores.seed==2026)].iloc[0]
            self.assertAlmostEqual(row.axis_mae,.16,places=8)
            report=m.finalize(
                root,{"source_sha256":"synthetic_test_only"},
                digest,self.panel)
            self.assertEqual(report["status"],"COMPLETE_MAIN_DEV_ANALYSIS")
            self.assertEqual(report["trained_length_seed_jobs"],200)
            self.assertEqual(report["operational_lookback_quarters"],
                             report["dev_selected_lookback_quarters"])
            self.assertEqual(report["original_reference_lookback_quarters"],12)
            self.assertFalse(report["train_internal_cv"])
            self.assertEqual(report["dev_policy_selection"]["fit_data"],
                             "TRAIN_2006Q1_2019Q4")
            self.assertEqual(
                len(pd.read_csv(root/"seed_scores.csv")),200)
            self.assertEqual(
                len(pd.read_csv(root/"lookback_summary.csv")),20)
            selected=m.json_read(root/"lookback_dev_selection.json")
            self.assertEqual(selected["reference_lookback_quarters"],12)
            self.assertEqual(selected["selection_rule"],m.DEV_SELECTION_RULE)
            self.assertEqual(selected["selected_lookback_quarters"],
                             int(pd.read_csv(root/"lookback_summary.csv").
                                 sort_values(["axis_mae","ici_mae",
                                              "input_quarters"]).iloc[0].
                                 input_quarters))
            self.assertNotIn("statistical_design",report)
            self.assertNotIn("omnibus",report)
            self.assertNotIn("statistical_design",
                             m.json_read(root/"run_manifest.json"))
            self.assertFalse(hasattr(m,"inferential_statistics"))
            self.assertFalse(hasattr(m,"holm_familywise_pvalues"))
            expected={
                "S05_01_input_length_mae.png",
                "S05_02_axes_and_ici.png",
                "S05_03_difference_from_12q.png",
                "S05_04_years_and_data_support.png"}
            actual={p.name for p in (root/"plots").glob("*.png")}
            self.assertEqual(actual,expected)
            for figure in (root/"plots").glob("*.png"):
                self.assertGreater(figure.stat().st_size,10000)
            self.assertFalse(report["test_opened"])

    def test_real_dev_selection_is_lookback_one_without_tests(self):
        selected=m.json_read(m.RESULTS/"lookback_dev_selection.json")
        self.assertEqual(selected["selected_lookback_quarters"],1)
        self.assertEqual(selected["reference_lookback_quarters"],12)
        self.assertEqual(selected["selection_rule"],m.DEV_SELECTION_RULE)
        self.assertFalse(selected["result_is_independent_validation"])
        self.assertFalse(selected["downstream_policy_applied"])


if __name__=="__main__":
    unittest.main()
