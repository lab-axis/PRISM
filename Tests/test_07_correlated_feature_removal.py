"""Stage07 L1 B-MTGNN, ten-seed compact pruning and leakage regression."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
from pipeline_support import require_current_results

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_prism07_greedy_unit_tests",ROOT/"Sources/07_correlated_feature_removal.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class GreedyLogicTests(unittest.TestCase):
    def test_true_compact_can_win_even_when_masked_diagnostic_fails(self):
        # A real compact model must be evaluated regardless of how the
        # 157-slot masked surrogate performs. It is not a valid gate.
        origin=m.Subset("no_removal",None,tuple(range(157)),(),
                        pd.DataFrame(),0)
        smaller=m.Subset("greedy_0.995",.995,
                         tuple(range(156)),(156,),pd.DataFrame(),1)
        second=m.Subset("greedy_0.990",.990,
                        tuple(range(155)),(155,156),pd.DataFrame(),2)
        base=dict(axis_mae=2.,ici_mae=1.6,
                  **{f"{year}_mae":2. for year in (2020,2021,2022)})
        trial=dict(
            policy="greedy_0.995",kept_count=156,axis_mae=1.9,
            ici_mae=1.58,
            **{"2020_mae":1.9,"2021_mae":1.9,"2022_mae":2.05})
        weaker=dict(
            policy="greedy_0.990",kept_count=155,axis_mae=2.1,
            ici_mae=1.59,
            **{"2020_mae":2.1,"2021_mae":2.1,"2022_mae":2.1})
        winner,score=m.select_compact_candidate(
            (origin,smaller,second),[trial,weaker],base)
        self.assertIs(winner,smaller)
        self.assertAlmostEqual(score["axis_mae"],1.9)
        baseline,score=m.select_compact_candidate(
            (origin,smaller,second),[
                dict(trial,axis_mae=2.2),weaker],base)
        self.assertIs(baseline,origin)
        self.assertEqual(score,base)

    def test_six_candidates_and_single_0_157_baseline(self):
        self.assertEqual(m.THRESHOLDS,(None,.995,.990,.980,.950,.900))
        self.assertEqual(len(m.THRESHOLDS),6)
        self.assertEqual(m.FEATURE_ROLES,(
            (0,42,"target_history"),(42,66,"industry_source"),
            (66,157,"global_source")))
        self.assertFalse(hasattr(m,"LOOKBACK"))
        self.assertEqual(m.HORIZON,1)
        self.assertEqual(m.SEEDS,tuple(range(2026,2036)))

    def test_transitive_chain_is_not_a_cluster(self):
        # A ↔ B and B ↔ C are highly correlated, but A ↔ C is not.
        # Fixed greedy order A,B,C MUST keep A and C, dropping only B.
        rho=np.array([[1.,.997,.5],
                      [.997,1.,.997],
                      [.5,.997,1.]])
        audit=m.CorrelationAudit(
            rho=rho,
            lower_abs=np.array([[1.,.996,.45],
                                [.996,1.,.996],
                                [.45,.996,1.]]),
            sign_consistency=np.ones((3,3)),
            support=np.full((3,3),56,int),
            observed_fraction=np.array([1.,.9,.8]),
            draws=40)
        result=m.greedy_prune(audit,["A","B","C"],.995)
        self.assertEqual(result.kept,(0,2))
        self.assertEqual(result.dropped,(1,))
        self.assertEqual(result.mapping.loc[
            result.mapping.feature.eq("B"),"representative"].iloc[0],"A")
        self.assertEqual(result.stable_edge_count,2)

    def test_semantically_different_channels_are_never_cross_pruned(self):
        n=157
        rho=np.eye(n)
        rho[1,66]=rho[66,1]=1.
        support=np.full((n,n),56,int)
        audit=m.CorrelationAudit(
            rho=rho,lower_abs=np.abs(rho).copy(),
            sign_consistency=np.ones_like(rho),
            support=support,
            observed_fraction=np.ones(n),
            draws=40)
        edges=m.stable_edges(audit,.995)
        self.assertFalse(edges[1,66])
        subset=m.greedy_prune(audit,[f"f{i}" for i in range(n)],.995)
        self.assertEqual(len(subset.kept),157)
        self.assertEqual(len(subset.dropped),0)

    def test_conservative_policy_gates(self):
        base=dict(axis_mae=2.,ici_mae=1.7,
                  **{"2020_mae":2.,"2021_mae":2.,"2022_mae":2.})
        good=dict(axis_mae=1.8,ici_mae=1.69,
                  **{"2020_mae":1.7,"2021_mae":1.8,"2022_mae":2.1})
        self.assertTrue(m.selection_safeguards(good,base)["qualified"])
        bad_ici=dict(good,ici_mae=1.71)
        self.assertFalse(m.selection_safeguards(bad_ici,base)["qualified"])
        bad_year=dict(good,**{"2020_mae":2.1,"2021_mae":2.1})
        self.assertFalse(m.selection_safeguards(bad_year,base)["qualified"])
        worse=dict(good,axis_mae=2.1)
        self.assertFalse(m.selection_safeguards(worse,base)["qualified"])


class OriginalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_current_results(6)
        torch.set_num_threads(4)
        cls.ctx=m.upstream()
        cls.panel=cls.ctx["panel"]
        cls.raw_train=cls.panel.x[:56]
        cls.audit=m.assess_train_correlations(
            cls.raw_train,draws=3,seed=20261013)

    def test_upstream_04_06_locked(self):
        ctx=self.ctx
        self.assertEqual(ctx["models"],("B-MTGNN",))
        self.assertEqual(ctx["epochs"],75)
        self.assertEqual(ctx["original_forecasts"].shape,(10,12,59,3))
        self.assertEqual(ctx["full_fold"]["train_x"].shape,
                         (len(ctx["m06"].train_origins()),ctx["lookback"],59,157))
        self.assertEqual(ctx["full_fold"]["val_x"].shape,(12,ctx["lookback"],59,157))
        self.assertEqual(ctx["panel"].x.shape,(68,59,157))
        self.assertEqual(ctx["panel"].y.shape,(68,59,3))
        self.assertNotIn("Data/03_integrated_dataset/raw_test.csv",
                         ctx["panel"].hashes)
        chosen=m.json_read(m.RESULT06/"selected_preprocessing.json")
        self.assertEqual(ctx["locked_policy"].id,chosen["policy_id"])

    def test_global_duplicate_quarter_not_pseudo_replication(self):
        raw=self.raw_train
        np.testing.assert_array_equal(
            raw[:,:,66:],
            np.broadcast_to(raw[:,:1,66:],raw[:,:,66:].shape))
        rho,support=m.correlation_views(raw)
        self.assertEqual(rho.shape,(157,157))
        self.assertTrue(np.isfinite(rho).all())
        self.assertEqual(support.shape,(157,157))
        self.assertLessEqual(int(support.max()),56)
        self.assertEqual(float(rho[0,0]),1.)
        self.assertEqual(float(rho[66,66]),1.)

    def test_train_only_no_distribution_leakage(self):
        x=self.panel.x.copy()
        x[56:,:,42:]=np.where(
            np.isfinite(x[56:,:,42:]),
            x[56:,:,42:]+1e9,np.nan)
        poison=self.ctx["m04"].Panel(
            x,self.panel.y+1e6,self.panel.quarters,
            self.panel.nodes,self.panel.names,
            self.panel.features,self.panel.train_count,
            self.panel.hashes)
        assert not np.array_equal(poison.x[56:],self.panel.x[56:])
        other=m.assess_train_correlations(
            poison.x[:56],draws=3,seed=20261013)
        for key in ("rho","support","lower_abs",
                    "sign_consistency","observed_fraction"):
            np.testing.assert_array_equal(
                getattr(self.audit,key),getattr(other,key))
        for threshold in m.THRESHOLDS:
            a=m.greedy_prune(self.audit,self.panel.features,threshold)
            b=m.greedy_prune(other,self.panel.features,threshold)
            self.assertEqual(a.kept,b.kept)
            self.assertEqual(a.dropped,b.dropped)

    def test_alternate_quarter_block_width_retains_valid_audit(self):
        alternate=m.assess_train_correlations(
            self.raw_train,draws=3,seed=20261013,width=8)
        self.assertEqual(alternate.rho.shape,(157,157))
        self.assertTrue(np.isfinite(alternate.lower_abs).all())
        self.assertTrue(np.isfinite(alternate.sign_consistency).all())

    def test_mask_and_compact_change_only_main_input(self):
        subset=m.greedy_prune(
            self.audit,self.panel.features,.90)
        self.assertGreater(len(subset.dropped),0)
        full=self.ctx["full_fold"]
        for kind in ("masked","compact"):
            altered=m.adapt_inputs(full,subset,kind)
            channel_count=157 if kind=="masked" else len(subset.kept)
            self.assertEqual(
                altered["train_x"].shape,(len(full["train_origins"]),self.ctx["lookback"],59,channel_count))
            self.assertEqual(
                altered["val_x"].shape,(12,self.ctx["lookback"],59,channel_count))
            for key in ("train_y","val_y","train_last","val_last",
                        "train_context","val_context",
                        "train_structure","val_structure",
                        "y_center","y_scale",
                        "train_origins","val_origins"):
                np.testing.assert_array_equal(altered[key],full[key])
            if kind=="masked":
                for key in (
                        "train_x","train_mask","val_x","val_mask"):
                    self.assertTrue(
                        np.all(altered[key][...,list(subset.dropped)]==0))
                np.testing.assert_array_equal(
                    altered["train_x"][...,list(subset.kept)],
                    full["train_x"][...,list(subset.kept)])
            else:
                np.testing.assert_array_equal(
                    altered["train_x"],
                    full["train_x"][...,list(subset.kept)])
        self.assertNotEqual(
            m.dataset_sha(m.adapt_inputs(full,subset,"masked")),
            m.dataset_sha(m.adapt_inputs(full,subset,"compact")))

    def test_original_ten_seed_scores_identical_to_selected_06(self):
        p=self.panel
        result=m.selected_baseline_scores(self.ctx)
        expected,indiv=self.ctx["m06"].seed_policy_metrics(
            self.ctx["original_forecasts"],p)
        self.assertAlmostEqual(result["axis_mae"],expected["axis_mae"])
        self.assertAlmostEqual(result["ici_mae"],expected["ici_mae"])
        self.assertEqual(len(indiv),10)
        self.assertAlmostEqual(result["axis_mae"],2.451713824742347)
        baseline=m.Subset(
            "no_removal",None,tuple(range(157)),(),pd.DataFrame(),0)
        masked=m.adapt_inputs(self.ctx["full_fold"],baseline,"masked")
        for key in ("train_x","val_x","train_mask","val_mask"):
            np.testing.assert_array_equal(
                masked[key],self.ctx["full_fold"][key])

    def test_compact_K_shares_initial_weights_and_fixed_core(self):
        subset=m.greedy_prune(
            self.audit,self.panel.features,.90)
        folded=m.adapt_inputs(self.ctx["full_fold"],subset,"compact")
        full_model=m.instantiate_subset(
            self.ctx,subset,2026,compact=False).eval()
        compact_model=m.instantiate_subset(
            self.ctx,subset,2026,compact=True).eval()
        k=len(subset.kept)
        self.assertEqual(compact_model.base.values.in_features,k)
        self.assertEqual(
            sum(p.numel() for p in full_model.parameters())-
            sum(p.numel() for p in compact_model.parameters()),
            2*self.ctx["m04"].STAGE04_ACTIVE_WIDTH*(157-k))
        torch.testing.assert_close(
            compact_model.base.values.weight,
            full_model.base.values.weight[:,list(subset.kept)])
        torch.testing.assert_close(
            compact_model.base.missing.weight,
            full_model.base.missing.weight[:,list(subset.kept)])
        for name,weight in compact_model.named_parameters():
            if name.startswith(("base.values.","base.missing.")):
                continue
            torch.testing.assert_close(
                weight,dict(full_model.named_parameters())[name])
        self.assertIsInstance(
            compact_model.base.core,self.ctx["m05"].FixedTemporalInput)
        graph=torch.eye(59)
        inputs={key:torch.tensor(folded["val_"+key][:1])
                for key in ("x","mask","last","structure","context")}
        with torch.inference_mode():
            prediction=compact_model(
                inputs["x"],inputs["mask"],graph,
                inputs["last"],inputs["structure"],inputs["context"])
        self.assertEqual(prediction.shape,(1,1,59,3))
        self.assertTrue(torch.isfinite(prediction).all())
        compact_model.train()
        train={key:torch.tensor(folded["train_"+key][:1])
               for key in ("x","mask","y","last","structure","context")}
        forecast=compact_model(
            train["x"],train["mask"],graph,
            train["last"],train["structure"],train["context"])
        loss=self.ctx["m04"].joint_axis_ici_loss(
            forecast,train["y"],
            torch.tensor(folded["y_scale"]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            torch.isfinite(compact_model.base.values.weight.grad).all())


@unittest.skipUnless(
    (m.RESULTS/"run_manifest.json").is_file() and
    m.json_read(m.RESULTS/"run_manifest.json").get("status")=="PASS",
    "07 full experiment has not finished")
class ProductionResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_current_results(7)
        cls.ctx=m.upstream()
        cls.root=m.RESULTS
        cls.subsets=pd.read_csv(cls.root/"subsets"/"candidate_counts.csv")
        cls.table=pd.read_csv(cls.root/"masked_threshold_comparison.csv")
        cls.lock=m.json_read(cls.root/"selected_features.json")
        cls.manifest=m.json_read(cls.root/"run_manifest.json")

    def test_complete_07_has_coherent_selected_features_and_no_clustering(self):
        self.assertEqual(self.manifest["status"],"PASS")
        self.assertEqual(self.lock["status"],"LOCKED")
        self.assertIn(self.lock["selected_policy"],[
            "no_removal",*[f"greedy_{threshold:.3f}" for threshold in m.THRESHOLDS[1:]]])
        self.assertIn(self.lock["selected_mode"],("original","compact"))
        k=self.lock["selected_feature_count"]
        self.assertEqual(self.lock["removed_feature_count"],157-k)
        self.assertEqual(len(self.lock["selected_indices"]),k)
        self.assertEqual(
            self.lock["selected_indices"],
            sorted(set(self.lock["selected_indices"])))
        self.assertFalse(self.lock["clustering_used"])
        self.assertTrue(self.lock["greedy_pairwise_only"])
        self.assertTrue(self.lock["main_input_only"])
        self.assertTrue(self.lock["prior_context_graph_and_last_unchanged"])
        self.assertFalse(self.lock["test_opened"])
        self.assertEqual(
            self.lock["selected_feature_names"],[
                self.ctx["panel"].features[i] for i in self.lock["selected_indices"]])
        self.assertEqual(self.manifest["signature"],
                         m.run_signature(m.run_contract(self.ctx)))
        self.assertFalse(hasattr(m,"cluster_features"))
        self.assertFalse(hasattr(m,"linkage"))

    def test_six_prespecified_real_subsets(self):
        self.assertEqual(len(self.subsets),6)
        self.assertEqual(int(self.subsets.iloc[0].kept_count),157)
        self.assertEqual(int(self.subsets.iloc[0].dropped_count),0)
        self.assertTrue(
            (self.subsets.kept_count+self.subsets.dropped_count==157).all())
        self.assertEqual(len(self.table),6)
        self.assertEqual(self.table.iloc[0].policy,"no_removal")
        self.assertAlmostEqual(self.table.iloc[0].axis_mae,
                               m.selected_baseline_scores(self.ctx)["axis_mae"])
        compact=pd.read_csv(self.root/"compact_confirmation.csv")
        self.assertEqual(len(compact),5)
        expected=compact[
            compact.qualified.astype(bool)].sort_values(
                ["axis_mae","removed_count","policy"],
                ascending=[True,False,True])
        if expected.empty:
            self.assertEqual(self.lock["selected_policy"],"no_removal")
        else:
            self.assertEqual(
                self.lock["selected_policy"],expected.iloc[0].policy)

    def test_full_masked_and_compact_trials_and_six_actual_plots(self):
        parents=self.root/"models"
        unique_masks=set()
        for label in self.subsets.label:
            groups=pd.read_csv(self.root/"subsets"/f"{label}.csv")
            unique_masks.add(tuple(groups.loc[
                groups.retained,"feature_index"].tolist()))
        expected=2*len(m.SEEDS)*(len(unique_masks)-1)
        self.assertEqual(len(list(parents.rglob("*.pt"))),expected)
        self.assertEqual(len(list(parents.rglob("*.npz"))),expected)
        self.assertEqual(len(list(parents.rglob("*.json"))),expected)
        self.assertEqual(len(list((self.root/"plots").glob("*.png"))),6)
        self.assertEqual(self.manifest["compact_candidates"],5)
        self.assertFalse(
            self.manifest["compact_confirmation"][
                "masked_diagnostic_used_for_compact_selection"])
        for name in self.lock["figures"]:
            self.assertGreater((self.root/name).stat().st_size,1000)
        self.assertGreaterEqual(
            self.manifest["distinct_masked_feature_sets"],1)
        audit=pd.read_csv(
            self.root/"correlations"/"block_sensitivity.csv")
        self.assertEqual(len(audit),25)
        self.assertTrue(
            (audit.loc[
                (audit.block_quarters==4)&
                (audit.bootstrap_draws==40),
                "different_feature_decisions"]==0).all())

    def test_each_variant_reproduces_seedwise_mae_and_parameter_count(self):
        compact=pd.read_csv(self.root/"compact_confirmation.csv")
        for mode,table in (("masked",self.table),("compact",compact)):
            for row in table.itertuples(index=False):
                if row.kept_count==157:
                    continue
                original=getattr(row,"same_subset_as",None)
                source=original if isinstance(original,str) else row.policy
                preds=[]
                for seed in m.SEEDS:
                    stem=f"B-MTGNN__s{seed}"
                    base=self.root/"models"/(source+"__"+mode)
                    with np.load(base/(stem+".npz"),allow_pickle=False) as z:
                        forecast=z["prediction"].copy()
                    self.assertEqual(forecast.shape,(12,59,3))
                    meta=m.json_read(base/(stem+".json"))
                    self.assertEqual(meta["epoch"],75)
                    self.assertEqual(meta["parameters"],row.parameters)
                    self.assertEqual(meta["prediction_sha256"],
                                     m.sha(base/(stem+".npz")))
                    self.assertFalse(meta["test_opened"])
                    preds.append(forecast)
                metrics,_=self.ctx["m06"].seed_policy_metrics(
                    np.stack(preds),self.ctx["panel"])
                with self.subTest(candidate=row.policy,mode=mode):
                    self.assertAlmostEqual(
                        metrics["axis_mae"],row.axis_mae,places=11)
                    self.assertAlmostEqual(
                        metrics["ici_mae"],row.ici_mae,places=11)
                    for year in (2020,2021,2022):
                        self.assertAlmostEqual(
                            metrics[f"{year}_mae"],
                            table.loc[
                                table.policy.eq(row.policy),
                                f"{year}_mae"].iloc[0],places=11)

    def test_selected_original_features_and_names_are_complete(self):
        indices=self.lock["selected_indices"]
        with np.load(
            self.root/"selected_feature_indices.npz",
            allow_pickle=False) as z:
            np.testing.assert_array_equal(z["indices"],np.asarray(indices))
            self.assertEqual(z["original_count"],157)
            np.testing.assert_array_equal(
                z["names"],np.asarray([
                    self.ctx["panel"].features[i] for i in indices]))
        selected=pd.read_csv(self.root/"selected_features.csv")
        dropped=pd.read_csv(self.root/"removed_features.csv")
        self.assertEqual(len(selected),len(indices))
        self.assertEqual(len(dropped),157-len(indices))
        self.assertEqual(selected.original_index.tolist(),indices)
        self.assertEqual(
            selected.feature.tolist(),[
                self.ctx["panel"].features[i] for i in indices])
        with np.load(self.root/"correlations"/"train_spearman_bootstrap.npz",
                     allow_pickle=False) as z:
            self.assertEqual(z["rho"].shape,(157,157))
            self.assertEqual(int(z["draws"]),m.BOOTSTRAPS)


if __name__=="__main__":
    unittest.main()
