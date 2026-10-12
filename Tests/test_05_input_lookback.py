"""PRISM S05: descriptive sensitivity only, NO model-selection output."""

from __future__ import annotations
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_prism05_supplementary_tests",ROOT/"Sources/05_input_lookback.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class SupplementaryContractTests(unittest.TestCase):
    def test_all_lengths_derived_from_declared_sample_support(self):
        d=m.study_design()
        self.assertEqual(d["upper"],20)
        self.assertEqual(d["lengths"],tuple(range(1,21)))
        self.assertEqual(d["early"].tolist(),list(range(20,40)))
        self.assertEqual(d["epoch"].tolist(),list(range(40,48)))
        self.assertEqual(d["refit"].tolist(),list(range(20,48)))
        self.assertEqual(d["measure"].tolist(),list(range(48,56)))
        self.assertEqual(m.ORIGINAL_LOOKBACK,12)
        self.assertGreaterEqual(len(d["early"]),max(d["lengths"]))

    def test_no_longer_any_length_selection_or_final_model_optimization(self):
        for obsolete in (
                "select_length","describe_sensitivity",
                "final_one","save_final","get_final",
                "archive_legacy_selected_outputs",
                "train_only_lookback_selection"):
            self.assertFalse(hasattr(m,obsolete),obsolete)

    def test_upstream_stage04_predefined_frozen_contract(self):
        locked=m.original_04()
        self.assertEqual(locked.mode,"structural")
        self.assertEqual(locked.seeds,tuple(m.stage04().DEFAULT_SEEDS))
        self.assertEqual(locked.models,(
            "B-MTGNN","APPNP-TCN","GraphConv-TCN",
            "GAT-TCN","GraphTransformer-TCN"))
        self.assertEqual(locked.weights,(.2,)*5)
        self.assertEqual(m.stage04().LOOKBACK,12)
        self.assertTrue((m.STAGE04_RESULTS/"ensemble_predictions.npz").is_file())


class FixedInputsAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.panel=m.stage04().load_panel(ROOT)
        cls.engine=m.Inputs(cls.panel)
        cls.design=m.study_design()

    def test_59_industries_157_features_and_test_closed(self):
        p=self.panel
        self.assertEqual(p.x.shape,(68,59,157))
        self.assertEqual(p.y.shape,(68,59,3))
        self.assertEqual(len(p.features)-115,42)
        self.assertNotIn("Data/03_integrated_dataset/raw_test.csv",p.hashes)

    def test_same_training_origins_all_lengths(self):
        for L in (1,8,12,20):
            a=self.engine.fold(
                L,40,self.design["early"],self.design["epoch"])
            b=self.engine.fold(
                L,48,self.design["refit"],self.design["measure"])
            self.assertEqual(a["train_x"].shape,(20,L,59,157))
            self.assertEqual(a["val_x"].shape,(8,L,59,157))
            self.assertEqual(b["train_x"].shape,(28,L,59,157))
            self.assertEqual(b["val_y"].shape,(8,1,59,3))

    def test_original_stage04_normalizer_and_asof_anchor(self):
        p=self.panel
        fold=self.engine.fold(
            12,48,self.design["refit"],self.design["measure"])
        old=m.stage04().prepare_three_axis_fold(
            p,self.engine.prior,48,48,56,"standard","mask")
        for k in ("x_center","x_scale","y_center","y_scale",
                  "val_x","val_mask","val_y","val_last","val_context",
                  "val_structure"):
            with self.subTest(name=k):
                np.testing.assert_array_equal(fold[k],old[k])

    def test_current_target_and_unavailable_predictors_do_not_enter_input(self):
        p=self.panel
        before=self.engine.fold(
            12,48,self.design["refit"],self.design["measure"])
        # The prediction origin 2018Q1 is withheld from inputs. Only the
        # historic as-of row-t target proxies are permitted.
        after_x=p.x.copy()
        after_y=p.y.copy()
        after_x[48:,:,42:]+=9999
        after_y[48:]+=9999
        mutated=m.stage04().Panel(
            after_x,after_y,p.quarters,p.nodes,p.names,p.features,
            p.train_count,p.hashes)
        self.engine.p=mutated
        try:
            changed=self.engine.fold(
                12,48,self.design["refit"],self.design["measure"])
        finally:
            self.engine.p=p
        for key in (
                "train_x","train_y","val_mask",
                "val_last","val_context","val_structure"):
            np.testing.assert_array_equal(before[key],changed[key])
        # The FIRST 2018Q1 forecast cannot see its own unknown X[48].
        np.testing.assert_array_equal(
            before["val_x"][0],changed["val_x"][0])
        # But 2018Q2/Q3 CAN see previous observed 2018Q1 economic X.
        self.assertFalse(np.array_equal(
            before["val_x"][1:],changed["val_x"][1:]))
        self.assertFalse(np.array_equal(before["val_y"],changed["val_y"]))

    def test_network_training_still_runs_without_result_publication(self):
        g=m.original_04()
        name=g.models[0]
        L=12
        early=self.engine.fold(L,40,self.design["early"],self.design["epoch"])
        with patch.object(m.stage04(),"STAGE04_ACTIVE_EPOCHS",2):
            epoch=m.train_network(
                self.engine,name,2026,L,g.mode,
                early,40,torch.device("cpu"))
        self.assertIn(epoch["best_epoch"],(1,2))
        second=self.engine.fold(
            L,48,self.design["refit"],self.design["measure"])
        pred=m.train_network(
            self.engine,name,2026,L,g.mode,
            second,48,torch.device("cpu"),
            epochs=epoch["best_epoch"])["prediction"]
        self.assertEqual(pred.shape,(8,59,3))
        self.assertTrue(np.isfinite(pred).all())


class CachedSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=m.stage04().load_panel(ROOT)
        cls.lock=m.original_04()
        cls.d=m.study_design()

    def test_complete_original_500_model_training_trials_untouched(self):
        root=m.RESULTS
        self.assertEqual(len(list((root/"trials").glob("*.json"))),500)
        self.assertEqual(len(list((root/"trials").glob("*.npz"))),500)
        self.assertEqual(m.read_json(root/"run_manifest.json")[
            "training_trials_sha256"],m.digest_trial_files(root))

    def test_all_500_cached_models_match_reaggregated_sensitivity(self):
        root=m.RESULTS
        signature=m.training_fingerprint(root,self.lock,self.panel,self.d)
        curve=m.score_500_trials(
            root,signature,self.lock,self.panel,self.d)
        recorded=pd.read_csv(root/"supplementary_sensitivity.csv")
        pd.testing.assert_frame_equal(
            curve.reset_index(drop=True),
            recorded.reset_index(drop=True),check_exact=False,atol=1e-12)
        self.assertEqual(len(curve),20)
        row12=curve.loc[curve.input_quarters.eq(12)].iloc[0]
        self.assertAlmostEqual(float(row12.axis_mae),0.9983377548536462)

    def test_descriptive_minimum_is_not_a_deployed_policy(self):
        report=m.read_json(m.RESULTS/"supplementary_report.json")
        self.assertEqual(report["status"],"PASS_SUPPLEMENTARY")
        self.assertEqual(report["operational_lookback"],12)
        self.assertEqual(report["descriptive_smallest_mae_length"],11)
        self.assertTrue(report["analysis_not_a_model_policy"])
        self.assertFalse(report["model_training_executed_during_rebuild"])
        self.assertFalse(report["test_opened"])
        self.assertFalse(report["statistical_equivalence_not_claimed"] is False)

    def test_old_model_selection_outputs_actually_deleted(self):
        root=m.RESULTS
        expected={
            "trials","plots","supplementary_sensitivity.csv",
            "supplementary_report.json","run_manifest.json"}
        self.assertEqual({p.name for p in root.iterdir()},expected)
        self.assertFalse(list(root.rglob("*.pt")))
        self.assertEqual(len(list((root/"plots").glob("*.png"))),4)
        for old in (
                "final","selected_input_lookback.json",
                "train_only_lookback_selection.json",
                "train_only_lookback_sensitivity.csv",
                "frozen_dev_predictions.npz"):
            self.assertFalse((root/old).exists())
        self.assertFalse(any(
            "archive" in f.name.lower() for f in root.rglob("*")))

    def test_rebuild_on_isolated_copy_never_calls_training(self):
        src=m.RESULTS
        with tempfile.TemporaryDirectory(prefix="prism05_test_rebuild_") as tmp:
            target=Path(tmp)/"results"
            target.mkdir()
            shutil.copytree(src/"trials",target/"trials")
            shutil.copy2(src/"run_manifest.json",target/"run_manifest.json")
            digest=m.digest_trial_files(target)
            # Simulate old detritus; all non-training leftovers must vanish.
            (target/"final").mkdir()
            (target/"final"/"L11__old_model.pt").write_bytes(b"obsolete")
            (target/"selected_input_lookback.json").write_text(
                '{"selected_lookback":11}',encoding="utf-8")
            with patch.object(m,"train_network",
                              side_effect=AssertionError("NO TRAINING ALLOWED")):
                result=m.rebuild(target,self.lock,self.panel,self.d)
            self.assertEqual(result["operational_lookback"],12)
            self.assertFalse((target/"final").exists())
            self.assertFalse((target/"selected_input_lookback.json").exists())
            self.assertEqual(m.digest_trial_files(target),digest)
            self.assertEqual(len(list((target/"plots").glob("*.png"))),4)


if __name__=="__main__":
    unittest.main()
