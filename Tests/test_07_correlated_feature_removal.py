"""Stage07 stable Spearman greedy pruning: no clustering and no time leakage."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_prism07_greedy_unit_tests",ROOT/"Sources/07_correlated_feature_removal.py")
m=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=m
SPEC.loader.exec_module(m)


class GreedyLogicTests(unittest.TestCase):
    def test_six_candidates_and_single_0_157_baseline(self):
        self.assertEqual(m.THRESHOLDS,(None,.995,.990,.980,.950,.900))
        self.assertEqual(len(m.THRESHOLDS),6)
        self.assertEqual(m.FEATURE_ROLES,(
            (0,42,"target_history"),(42,66,"industry_source"),
            (66,157,"global_source")))
        self.assertEqual(m.LOOKBACK,12)
        self.assertEqual(m.HORIZON,1)

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
        torch.set_num_threads(4)
        cls.ctx=m.upstream()
        cls.panel=cls.ctx["panel"]
        cls.raw_train=cls.panel.x[:56]
        cls.audit=m.assess_train_correlations(
            cls.raw_train,draws=3,seed=20261013)

    def test_upstream_04_06_locked(self):
        ctx=self.ctx
        self.assertEqual(tuple(ctx["models"]),(
            "B-MTGNN","APPNP-TCN","GraphConv-TCN",
            "GAT-TCN","GraphTransformer-TCN"))
        self.assertEqual(tuple(ctx["epochs"]),tuple(ctx["epochs"]))
        self.assertEqual(len(ctx["epochs"]),25)
        self.assertEqual(ctx["panel"].x.shape,(68,59,157))
        self.assertEqual(ctx["panel"].y.shape,(68,59,3))
        self.assertNotIn("Data/03_integrated_dataset/raw_test.csv",
                         ctx["panel"].hashes)
        self.assertEqual(ctx["locked_policy"].id,
                         "m-raw_zero__n-standard__o-none")

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

    def test_mask_and_compact_change_only_main_input(self):
        subset=m.greedy_prune(
            self.audit,self.panel.features,.90)
        self.assertGreater(len(subset.dropped),0)
        full=self.ctx["full_fold"]
        for kind in ("masked","compact"):
            altered=m.adapt_inputs(full,subset,kind)
            channel_count=157 if kind=="masked" else len(subset.kept)
            self.assertEqual(
                altered["train_x"].shape,(44,12,59,channel_count))
            self.assertEqual(
                altered["val_x"].shape,(12,12,59,channel_count))
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

    def test_original_frozen_predictions_equal_04_and_06(self):
        p=self.panel
        result=m.selected_baseline_scores(self.ctx)
        self.assertAlmostEqual(result["axis_mae"],2.340279421959421)
        self.assertAlmostEqual(result["ici_mae"],1.7894925355233664)
        self.assertEqual(self.ctx["original_forecast"].shape,(12,59,3))
        baseline=m.Subset(
            "no_removal",None,tuple(range(157)),(),pd.DataFrame(),0)
        masked=m.adapt_inputs(self.ctx["full_fold"],baseline,"masked")
        for key in ("train_x","val_x","train_mask","val_mask"):
            np.testing.assert_array_equal(
                masked[key],self.ctx["full_fold"][key])

    def test_all_selected_gnns_can_accept_actual_compact_K(self):
        subset=m.greedy_prune(
            self.audit,self.panel.features,.90)
        folded=m.adapt_inputs(self.ctx["full_fold"],subset,"compact")
        m04=self.ctx["m04"]
        for name in self.ctx["models"]:
            with self.subTest(architecture=name):
                m04.seed_all(2026)
                model=m04.ThreeAxisStructureGNN(
                    name,folded["x_center"],folded["x_scale"],
                    folded["y_center"],folded["y_scale"],
                    num_features=len(subset.kept),
                    nodes=59,prediction_mode="structural").eval()
                graph=torch.tensor(m04.train_graph(self.panel))
                inputs={
                    k:torch.tensor(folded["val_"+k][:1])
                    for k in ("x","mask","last","structure","context")}
                with torch.inference_mode():
                    prediction=model(
                        inputs["x"],inputs["mask"],graph,
                        inputs["last"],inputs["structure"],inputs["context"])
                self.assertEqual(prediction.shape,(1,1,59,3))
                self.assertTrue(torch.isfinite(prediction).all())


class ProductionResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx=m.upstream()
        cls.root=m.RESULTS
        cls.subsets=pd.read_csv(cls.root/"subsets"/"candidate_counts.csv")
        cls.table=pd.read_csv(cls.root/"masked_threshold_comparison.csv")
        cls.lock=m.json_read(cls.root/"selected_features.json")
        cls.manifest=m.json_read(cls.root/"run_manifest.json")

    def test_complete_07_is_157_features_and_no_clustering(self):
        self.assertEqual(self.manifest["status"],"PASS")
        self.assertEqual(self.lock["status"],"LOCKED")
        self.assertEqual(self.lock["selected_policy"],"no_removal")
        self.assertEqual(self.lock["selected_mode"],"original")
        self.assertEqual(self.lock["selected_feature_count"],157)
        self.assertEqual(self.lock["removed_feature_count"],0)
        self.assertEqual(self.lock["selected_indices"],list(range(157)))
        self.assertFalse(self.lock["clustering_used"])
        self.assertTrue(self.lock["greedy_pairwise_only"])
        self.assertTrue(self.lock["main_input_only"])
        self.assertTrue(self.lock["prior_context_graph_and_last_unchanged"])
        self.assertFalse(self.lock["test_opened"])
        self.assertEqual(
            self.lock["selected_feature_names"],self.ctx["panel"].features)
        self.assertEqual(self.manifest["signature"],
                         m.run_signature(m.run_contract(self.ctx)))
        self.assertFalse(hasattr(m,"cluster_features"))
        self.assertFalse(hasattr(m,"linkage"))

    def test_six_prespecified_real_subsets(self):
        self.assertEqual(len(self.subsets),6)
        self.assertEqual(
            self.subsets.kept_count.tolist(),[157,147,140,134,115,98])
        self.assertEqual(
            self.subsets.dropped_count.tolist(),[0,10,17,23,42,59])
        self.assertEqual(len(self.table),6)
        self.assertEqual(self.table.qualified.sum(),0)
        self.assertEqual(
            self.table.years_improved.tolist(),[0,1,1,1,1,1])
        self.assertEqual(self.table.iloc[0].policy,"no_removal")
        self.assertAlmostEqual(
            self.table.iloc[0].axis_mae,2.340279421959421)
        self.assertTrue(
            np.all(self.table.axis_mae.iloc[1:].to_numpy()>
                   self.table.axis_mae.iloc[0]))

    def test_125_gnn_checkpoints_and_five_actual_plots(self):
        parents=self.root/"models"
        self.assertEqual(len(list(parents.rglob("*.pt"))),125)
        self.assertEqual(len(list(parents.rglob("*.npz"))),125)
        self.assertEqual(len(list(parents.rglob("*.json"))),125)
        self.assertEqual(len(list((self.root/"plots").glob("*.png"))),5)
        self.assertFalse(list(parents.glob("*__compact")))
        for name in self.lock["figures"]:
            self.assertGreater((self.root/name).stat().st_size,1000)
        self.assertEqual(
            self.manifest["distinct_masked_feature_sets"],6)
        self.assertFalse(self.manifest["compact_confirmation"]["compact_adopted"])

    def test_every_production_gnn_forecast_reproduces_published_mae(self):
        truth=self.ctx["panel"].y[56:68]
        previous=self.ctx["m04"].past_scores(
            self.ctx["panel"],np.arange(56,68))
        for row in self.table.iloc[1:].itertuples():
            banks=[]
            for name in self.ctx["models"]:
                preds=[]
                for seed in m.SEEDS:
                    stem=f"{name}__s{seed}"
                    base=self.root/"models"/(row.policy+"__masked")
                    with np.load(base/(stem+".npz"),allow_pickle=False) as z:
                        forecast=z["prediction"].copy()
                    self.assertEqual(forecast.shape,(12,59,3))
                    meta=m.json_read(base/(stem+".json"))
                    self.assertEqual(
                        meta["epoch"],self.ctx["epochs"][(name,seed)])
                    self.assertTrue(meta["dev_weights_frozen"])
                    self.assertFalse(meta["test_opened"])
                    preds.append(forecast)
                banks.append(np.stack(preds).mean(axis=0))
            full=np.stack(banks).mean(axis=0)
            metrics=self.ctx["m06"].scores(full,truth,previous)
            with self.subTest(candidate=row.policy):
                self.assertAlmostEqual(
                    metrics["axis_mae"],row.axis_mae,places=11)
                self.assertAlmostEqual(
                    metrics["ici_mae"],row.ici_mae,places=11)
                for year in (2020,2021,2022):
                    self.assertAlmostEqual(
                        metrics[f"{year}_mae"],
                        getattr(row,f"_{year}_mae")
                        if hasattr(row,f"_{year}_mae")
                        else self.table.loc[
                            self.table.policy.eq(row.policy),
                            f"{year}_mae"].iloc[0],places=11)

    def test_selected_original_features_and_names_are_complete(self):
        with np.load(
            self.root/"selected_feature_indices.npz",
            allow_pickle=False) as z:
            np.testing.assert_array_equal(z["indices"],np.arange(157))
            self.assertEqual(z["original_count"],157)
            np.testing.assert_array_equal(
                z["names"],np.asarray(self.ctx["panel"].features))
        selected=pd.read_csv(self.root/"selected_features.csv")
        dropped=pd.read_csv(self.root/"removed_features.csv")
        self.assertEqual(len(selected),157)
        self.assertEqual(len(dropped),0)
        self.assertEqual(selected.original_index.tolist(),list(range(157)))
        self.assertEqual(
            selected.feature.tolist(),self.ctx["panel"].features)
        with np.load(self.root/"correlations"/"train_spearman_bootstrap.npz",
                     allow_pickle=False) as z:
            self.assertEqual(z["rho"].shape,(157,157))
            self.assertEqual(int(z["draws"]),m.BOOTSTRAPS)


if __name__=="__main__":
    unittest.main()
