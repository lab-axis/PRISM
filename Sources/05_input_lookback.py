#!/usr/bin/env python3
"""PRISM 05 — MAIN EXPERIMENT: Dev-selected B-MTGNN lookback policy.

RESEARCH QUESTION
    Using the entire 2006–2019 Train period for preprocessing and all 36
    common supervised training origins 2011–2019, compare L=1..20 raw
    quarterly input windows on Dev 2020–2022 and select one length there.

DEV POLICY SELECTION (DESCRIPTIVE)
    For each L and seed, calculate the three-axis Dev MAE individually.
    Choose the L with the lowest average across the 10 matched seeds;
    ICI MAE then shorter L break ties. L=1 is the observed Dev selection.
    Report means, between-seed SD, per-axis/year errors and descriptive
    differences from original L=12, without p-values or confidence intervals.

INTERPRETATION
    Dev was already used to select Stage04's Structural formulation.
    L=1 is a Dev-selected modeling policy, NOT independently validated
    predictive superiority. L=12 is retained only as the original reference.
    Downstream stages are not modified by this code.

PROTOCOL — NO TRAIN CV, ROLLING, EXPANDING OR TRAIN-INTERNAL HOLDOUT
    B-MTGNN Structural only; 10 matched seeds 2026–2035; the same
    67,190-parameter model and initialization across L; fixed 20-slot
    temporal core; 75 epochs and 30 MC-dropout draws. The same nine
    historical score features and economic prior remain for every L,
    so this compares raw feature-window length conditional on context,
    NOT the total amount of history available. Test is not accessed.

    py Sources/05_input_lookback.py --analyze   # report/plots only; no fit
    py Sources/05_input_lookback.py --status
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"Sources/05_input_lookback.py"
SOURCE04=ROOT/"Sources/04_bmtgnn_benchmark.py"
RESULT04=ROOT/"Results/04_bmtgnn_benchmark"
RESULTS=ROOT/"Results/05_input_lookback"
ORIGINAL_LOOKBACK=12
LENGTHS=tuple(range(1,21))
COMMON_TRAIN_ORIGINS=np.arange(20,56,dtype=int)
DEV_ORIGINS=np.arange(56,68,dtype=int)
SEEDS=tuple(range(2026,2036))
MODEL="B-MTGNN"
MODE="structural"
CORE_LOOKBACK=max(LENGTHS)
SCORE_NAMES=("profitability","productivity","growth")
DEV_SELECTION_RULE="MIN_10_SEED_MEAN_THREE_AXIS_MAE_THEN_ICI_THEN_SHORTER_L"


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path:Path,value:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".json.writing")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,
                               ensure_ascii=False),encoding="utf-8")
    os.replace(temp,path)


def save_npz(path:Path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".writing.npz")
    np.savez_compressed(temp,**arrays)
    os.replace(temp,path)


@lru_cache(maxsize=1)
def stage04():
    spec=importlib.util.spec_from_file_location("_prism05_full_train_04",SOURCE04)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def upstream():
    """Validate exactly the current frozen B-MTGNN Stage04 Dev contract."""
    m=stage04()
    panel=m.load_panel(ROOT)
    files=(RESULT04/"run_manifest.json",RESULT04/"dev_selection.json")
    if not all(x.is_file() for x in files):
        raise RuntimeError("Frozen B-MTGNN Stage04 Dev results required")
    manifest,lock=map(json_read,files)
    if (manifest.get("status")!="DEV_FROZEN"
            or manifest.get("completed_jobs")!=30
            or manifest.get("total_jobs")!=30
            or manifest.get("contract")!=m.benchmark_contract()
            or manifest["contract"].get("model")!=MODEL
            or manifest["contract"].get("reference_epoch")!=75
            or tuple(manifest["contract"].get("seeds",()))!=SEEDS
            or lock.get("model")!=MODEL
            or lock.get("selected_mode")!=MODE
            or lock.get("seed_count")!=len(SEEDS)
            or lock.get("source_sha256")!=manifest.get("training_source_sha256")
            or m.STAGE04_ACTIVE_EPOCHS!=75
            or panel.x.shape!=(68,59,157)
            or panel.train_count!=56):
        raise RuntimeError("Stage05 requires frozen 04 B-MTGNN Structural Dev")
    # The historic checkpoints keep their original training-source SHA;
    # plotting/analysis edits do not retrospectively alter it.
    m._read_completed_jobs(manifest["training_source_sha256"])
    return m,panel,lock,manifest


def study_design():
    if (len(LENGTHS)!=20 or not 1<=ORIGINAL_LOOKBACK<=20
            or len(COMMON_TRAIN_ORIGINS)!=36
            or DEV_ORIGINS.tolist()!=list(range(56,68))
            or COMMON_TRAIN_ORIGINS[-1]!=55):
        raise RuntimeError("Stage05 full Train / Dev design drift")
    return dict(
        input_lengths=list(LENGTHS),
        train_origin_indices=COMMON_TRAIN_ORIGINS.tolist(),
        dev_origin_indices=DEV_ORIGINS.tolist(),
        all_2006_2019_is_train=True,
        train_heldout_years=[],
        inner_validation=False,
        epoch_selection="NONE_FIXED_75",
        dev_used_for_main_policy_selection=True,
        train_internal_cv=False,
        training_strategy="SINGLE_FULL_TRAIN_FIT",
        same_model_parameters_all_lengths=True,
        same_initialization_per_seed=True,
        fixed_core_temporal_capacity=CORE_LOOKBACK,
        common_asof_9d_target_context=True,
        experiment="conditional_quarterly_feature_window_not_total_history",
        original_reference_L=ORIGINAL_LOOKBACK,
        selection_rule=DEV_SELECTION_RULE)


class Inputs:
    def __init__(self,panel):
        self.panel=panel
        self.m=stage04()
        # Use the same Train-only structural anchor as the frozen Stage04.
        self.prior,_=self.m.make_fold(panel)
        # B-MTGNN learns its own graph; it does not take a fitted adjacency.
        self.graph=torch.eye(self.m.NODE_COUNT)

    def fold(self,L):
        if L not in LENGTHS:
            raise ValueError("Stage05 lookback is in [1,20]")
        p=self.panel
        train=COMMON_TRAIN_ORIGINS
        dev=DEV_ORIGINS
        center,scale=self.m.fit_normalizer(p.x[:56],"standard")
        known=np.isfinite(p.x)
        x=((np.where(known,p.x,0.)-center)/scale).astype(np.float32)
        mask=(~known).astype(np.float32)
        yc=p.y[:56].mean((0,1))
        ys=p.y[:56].std((0,1))
        ys=np.where(ys>1e-8,ys,1.)
        norm_y=((p.y-yc)/ys).astype(np.float32)
        structural=((self.prior-yc)/ys).astype(np.float32)
        def pack(indices,prefix):
            return {
                prefix+"_x":np.stack([x[t-L:t] for t in indices]),
                prefix+"_mask":np.stack([mask[t-L:t] for t in indices]),
                prefix+"_y":np.stack([norm_y[t:t+1] for t in indices]),
                prefix+"_last":((
                    self.m.past_scores(p,indices)-yc)/ys).astype(np.float32),
                prefix+"_context":self.m.known_score_context(
                    p,indices,yc,ys),
                prefix+"_structure":structural[indices]}
        arrays=dict(
            **pack(train,"train"),**pack(dev,"val"),
            x_center=center.astype(np.float32),
            x_scale=scale.astype(np.float32),
            y_center=yc.astype(np.float32),
            y_scale=ys.astype(np.float32),
            train_origins=train.copy(),val_origins=dev.copy())
        if (arrays["train_x"].shape!=(36,L,59,157)
                or arrays["val_x"].shape!=(12,L,59,157)
                or not all(np.isfinite(v).all() for v in arrays.values())):
            raise RuntimeError("Stage05 Train/Dev array contract broken")
        return arrays


class FixedTemporalInput(torch.nn.Module):
    """Non-trainable latent left padding; B-MTGNN core always sees 20 slots.

    Padding AFTER the normalizer/context adapter is essential: padding raw
    features with zeros would create false context features and bias terms.
    Only genuinely observed L quarters enter the latent representation.
    """
    def __init__(self,backbone):
        super().__init__()
        self.backbone=backbone

    def forward(self,encoded,graph):
        L=encoded.shape[1]
        if not 1<=L<=CORE_LOOKBACK:
            raise ValueError("B-MTGNN input length must lie in 1..20")
        if L<CORE_LOOKBACK:
            pad=encoded.new_zeros(
                (encoded.shape[0],CORE_LOOKBACK-L,
                 encoded.shape[2],encoded.shape[3]))
            encoded=torch.cat((pad,encoded),dim=1)
        return self.backbone(encoded,graph)


def instantiate(m,name,seed,L,z,mode):
    """Keep EVERY parameter shape and initial tensor identical for all L.

    The original MTGNN temporal stack changes its kernel/norm dimensions at
    L=20. Fix its core width to 20 for the ENTIRE study, including L=12.
    B-MTGNN's core pads shorter observed sequences on the left with zeros.
    No different trainable resize/pooling adapter or L-specific head exists.
    This L=12 comparator is a newly trained common-support Stage05 control,
    not the saved Stage04 L=12 checkpoint.
    """
    if (name!=MODEL or mode!=MODE or L not in LENGTHS
            or z["train_x"].shape[1]!=L):
        raise ValueError("Stage05 requires B-MTGNN Structural and L=1..20")
    m.seed_all(seed)
    model=m.ThreeAxisStructureGNN(
        name,z["x_center"],z["x_scale"],
        z["y_center"],z["y_scale"],num_features=157,
        nodes=59,prediction_mode=mode)
    model.base=m.ForecastModel(
        name,157,59,m.STAGE04_ACTIVE_WIDTH,CORE_LOOKBACK,m.HORIZON)
    model.base.core=FixedTemporalInput(model.base.core)
    return model


def train_one(inputs,model_name,seed,L,device):
    m=inputs.m
    z=inputs.fold(L)
    net=instantiate(m,model_name,seed,L,z,"structural").to(device)
    graph=inputs.graph.to(device)
    ztorch={k:torch.as_tensor(z[k],device=device)
            for k in ("train_x","train_mask","train_y",
                      "train_last","train_context","train_structure",
                      "val_x","val_mask","val_last",
                      "val_context","val_structure")}
    optim=torch.optim.AdamW(
        net.parameters(),lr=m.FIXED_LEARNING_RATE,weight_decay=1e-4)
    scale=torch.as_tensor(z["y_scale"],device=device)
    for _ in range(m.STAGE04_ACTIVE_EPOCHS):
        net.train()
        optim.zero_grad(set_to_none=True)
        output=net(ztorch["train_x"],ztorch["train_mask"],graph,
                   ztorch["train_last"],ztorch["train_structure"],
                   ztorch["train_context"])
        cost=m.three_axis_weighted_huber(
            output,ztorch["train_y"],ztorch["train_last"],scale)
        if not torch.isfinite(cost):
            raise RuntimeError("Stage05 nonfinite full-Train training loss")
        cost.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),5.,error_if_nonfinite=True)
        optim.step()
    with m.isolated_snapshot_inference(net):
        estimate=m.mtgnn_bayesian_inference(
            net,ztorch,graph,model_name,seed,
            z["y_center"],z["y_scale"])["prediction"]
    del net,optim,ztorch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if estimate.shape!=(12,59,3) or not np.isfinite(estimate).all():
        raise RuntimeError("Stage05 Dev prediction invalid")
    return estimate


def run_contract(m,panel,lock,manifest):
    config=dict(
        stage="S05_MAIN_BMTGNN_CONTROLLED_LOOKBACK_COMPARISON",
        source_sha256=sha(SOURCE),
        stage04_source_sha256=sha(SOURCE04),
        stage04_manifest_sha256=sha(RESULT04/"run_manifest.json"),
        stage04_selection_sha256=sha(RESULT04/"dev_selection.json"),
        stage04_trained_source_sha256=manifest["training_source_sha256"],
        original_train_dev_sources=panel.hashes,
        model=MODEL,formulation=MODE,seeds=list(SEEDS),
        fixed_epochs=m.STAGE04_ACTIVE_EPOCHS,
        full_train_through="2019Q4",
        train_origin_count=36,
        dev_from="2020Q1",dev_end="2022Q4",
        no_train_internal_validation=True,
        withheld_train_years=[],
        all_train_years_used_for_model_fitting=True,
        lookback_grid=list(LENGTHS),
        fixed_bmtgnn_core_temporal_capacity=CORE_LOOKBACK,
        fixed_model_shape_every_length=True,
        stage04_12q_checkpoint_is_not_stage05_12q_control=True,
        context_history="FIXED_9D_ASOF_INCLUDES_LAG5_FOR_ALL_L",
        seed_scoring="PER_SEED_ORIGINAL_THREE_AXIS_MAE_THEN_AVERAGE",
        learned_graph="B_MTGNN_TRAINED_INTERNAL_ADJACENCY",
        operational_L=ORIGINAL_LOOKBACK,
        test_opened=False)
    digest=hashlib.sha256(
        json.dumps(config,sort_keys=True).encode()).hexdigest()
    return config,digest


def score_results(root,policy_sig,panel):
    truth=panel.y[DEV_ORIGINS]
    records=[]
    for L in LENGTHS:
        for seed in SEEDS:
            stem=f"L{L:02d}__{MODEL}__s{seed}"
            mpath=root/"trials"/(stem+".json")
            ppath=root/"trials"/(stem+".npz")
            if not mpath.is_file() or not ppath.is_file():
                raise RuntimeError(f"Missing Stage05 Dev trial {stem}")
            meta=json_read(mpath)
            if (meta.get("signature")!=policy_sig or
                    meta.get("train_internal_validation") is not False or
                    meta.get("L")!=L or meta.get("model")!=MODEL or
                    meta.get("seed")!=seed or
                    meta.get("fixed_epochs")!=75 or
                    meta.get("prediction_sha256")!=sha(ppath)):
                raise RuntimeError(f"Inconsistent Stage05 trial {stem}")
            with np.load(ppath,allow_pickle=False) as data:
                prediction=data["prediction"].copy()
            if (prediction.shape!=(12,59,3) or
                    not np.isfinite(prediction).all()):
                raise RuntimeError(f"Malformed Dev forecast {stem}")
            residual=prediction-truth
            row=dict(input_quarters=L,seed=seed,model=MODEL,
                     mode=MODE,axis_mae=float(np.abs(residual).mean()),
                     ici_mae=float(np.abs(residual.mean(-1)).mean()),
                     train_origin_count=36,dev_origin_count=12)
            for j,name in enumerate(SCORE_NAMES):
                row[name+"_mae"]=float(np.abs(residual[...,j]).mean())
            for y,year in enumerate((2020,2021,2022)):
                four=residual[4*y:4*(y+1)]
                row[f"{year}_mae"]=float(np.abs(four).mean())
                row[f"{year}_ici_mae"]=float(
                    np.abs(four.mean(-1)).mean())
            records.append(row)
    return pd.DataFrame(records).sort_values(
        ["input_quarters","seed"]).reset_index(drop=True)


def summarize_seed_scores(seed_scores):
    """Aggregate seed-level errors. NEVER score the seed-mean forecast."""
    expected={(length,seed) for length in LENGTHS for seed in SEEDS}
    actual=set(zip(seed_scores.input_quarters,seed_scores.seed))
    if len(seed_scores)!=200 or actual!=expected:
        raise RuntimeError("Exactly 20 lookbacks × 10 seeds required")
    baseline=seed_scores.loc[
        seed_scores.input_quarters==ORIGINAL_LOOKBACK].set_index("seed")
    rows=[]
    for L in LENGTHS:
        selected=seed_scores.loc[
            seed_scores.input_quarters==L].set_index("seed")
        if set(selected.index)!=set(baseline.index):
            raise RuntimeError("Unpaired seed set")
        row=dict(
            input_quarters=L,model=MODEL,mode=MODE,
            seed_count=len(SEEDS),train_origin_count=36,
            dev_origin_count=12,train_internal_validation=False,
            epoch=75,operational_12q=(L==ORIGINAL_LOOKBACK))
        for metric in ("axis_mae","ici_mae",
                       *(name+"_mae" for name in SCORE_NAMES),
                       *(f"{year}_{suffix}" for year in (2020,2021,2022)
                         for suffix in ("mae","ici_mae"))):
            val=selected[metric]
            row[metric]=float(val.mean())
            row[metric+"_seed_sd"]=float(val.std(ddof=1))
            difference=val-baseline[metric]
            row[metric+"_paired_delta_vs_12q"]=float(difference.mean())
            row[metric+"_paired_delta_sd"]=float(
                difference.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("input_quarters").reset_index(drop=True)


def finalize(root,contract,digest,panel):
    seed_scores=score_results(root,digest,panel)
    table=summarize_seed_scores(seed_scores)
    seed_scores.to_csv(root/"seed_scores.csv",
                       index=False,encoding="utf-8-sig")
    table.to_csv(root/"lookback_summary.csv",
                 index=False,encoding="utf-8-sig")
    selected_record=table.sort_values(
        ["axis_mae","ici_mae","input_quarters"]).iloc[0]
    selected=int(selected_record.input_quarters)
    images=root/"plots"
    images.mkdir(exist_ok=True)

    def save(fig,name):
        fig.tight_layout()
        fig.savefig(images/name,dpi=175,bbox_inches="tight")
        plt.close(fig)

    # Restore the original FOUR S05 figures and relative filenames.
    fig,ax=plt.subplots(figsize=(11.4,5.4))
    x=table.input_quarters
    ax.plot(x,table.axis_mae,"o-",lw=2,
            label="Ten-seed mean 3-axis MAE")
    ax.fill_between(
        x,table.axis_mae-table.axis_mae_seed_sd,
        table.axis_mae+table.axis_mae_seed_sd,
        alpha=.17,label="Between-seed ±1 SD (not CI)")
    ax.axvline(ORIGINAL_LOOKBACK,color="tab:orange",ls="--",
               label="Original reference L=12")
    ax.axvline(selected,color="tab:green",ls="-.",
               label=f"Dev selected L={selected}")
    ax.set(xlabel="Observed quarterly feature-input window L",
           ylabel="Dev 3-axis MAE",
           title="B-MTGNN lookback sensitivity | same model capacity")
    ax.set_xticks(LENGTHS)
    ax.grid(alpha=.2);ax.legend()
    save(fig,"S05_01_input_length_mae.png")

    fig,ax=plt.subplots(figsize=(11.4,5.4))
    for metric,label in (("profitability_mae","Profitability"),
                         ("productivity_mae","Productivity"),
                         ("growth_mae","Growth"),("ici_mae","ICI")):
        ax.plot(x,table[metric],"o-",markersize=3,label=label)
    ax.axvline(ORIGINAL_LOOKBACK,color="tab:orange",ls="--",
               label="Original L=12")
    ax.axvline(selected,color="tab:green",ls="-.",
               label=f"Dev selected L={selected}")
    ax.set(xlabel="Observed input quarters",ylabel="Dev MAE",
           title="Original three target axes and ICI across lookback")
    ax.set_xticks(LENGTHS)
    ax.grid(alpha=.2);ax.legend()
    save(fig,"S05_02_axes_and_ici.png")

    fig,ax=plt.subplots(figsize=(11.4,4.8))
    delta=table["axis_mae_paired_delta_vs_12q"]
    spread=table["axis_mae_paired_delta_sd"]
    ax.axhline(0.,color="black",ls=":")
    ax.errorbar(x,delta,yerr=spread,fmt="o-",capsize=3,
                label="Seed-paired MAE difference ±1 SD")
    ax.axvline(ORIGINAL_LOOKBACK,color="tab:orange",ls="--",
               label="Original reference L=12")
    ax.axvline(selected,color="tab:green",ls="-.",
               label=f"Dev selected L={selected}")
    ax.set(xlabel="Observed input quarters",
           ylabel="MAE(L) − MAE(L12)",
           title="Seed-paired sensitivity versus fixed L=12")
    ax.set_xticks(LENGTHS)
    ax.grid(alpha=.2);ax.legend()
    save(fig,"S05_03_difference_from_12q.png")

    fig,axes=plt.subplots(1,2,figsize=(14.2,5.0))
    for year in (2020,2021,2022):
        axes[0].plot(x,table[f"{year}_mae"],"o-",markersize=3,
                     label=str(year))
    axes[0].axvline(ORIGINAL_LOOKBACK,ls="--",color="tab:orange")
    axes[0].axvline(selected,ls="-.",color="tab:green",
                   label=f"Dev selected L={selected}")
    axes[0].set(title="Year-specific three-axis Dev MAE",
                xlabel="Observed input quarters",ylabel="MAE")
    axes[0].grid(alpha=.2);axes[0].legend()
    axes[1].plot(x,[len(COMMON_TRAIN_ORIGINS)]*len(x),
                 "o-",label="Train target origins")
    axes[1].plot(x,[len(DEV_ORIGINS)]*len(x),
                 "s--",label="Dev target origins")
    axes[1].set_ylim(0,42)
    axes[1].set(title="Identical target-origin support at every L",
                xlabel="Observed input quarters",ylabel="Origins")
    axes[1].grid(alpha=.2);axes[1].legend()
    save(fig,"S05_04_years_and_data_support.png")

    selection=dict(
        model=MODEL,mode=MODE,
        selected_lookback_quarters=selected,
        reference_lookback_quarters=ORIGINAL_LOOKBACK,
        selection_data="DEV_2020Q1_2022Q4",
        fit_data="TRAIN_2006Q1_2019Q4",
        comparison_train_target_origins="2011Q1_2019Q4",
        selection_rule=DEV_SELECTION_RULE,
        selected_dev_mean_three_axis_mae=float(selected_record.axis_mae),
        selected_dev_mean_ici_mae=float(selected_record.ici_mae),
        seed_count=len(SEEDS),
        result_is_independent_validation=False,
        prior_dev_exposure=True,
        downstream_policy_applied=False)
    json_write(root/"lookback_dev_selection.json",selection)
    report=dict(
        stage="S05",status="COMPLETE_MAIN_DEV_ANALYSIS",
        experiment_class="MAIN_CONTROLLED_LOOKBACK_COMPARISON",
        policy_selection="TEN_SEED_MEAN_DEV_THREE_AXIS_MAE",
        epoch_selection="NONE_FIXED_75",
        input_lengths=list(LENGTHS),
        model=MODEL,mode=MODE,
        seed_count=len(SEEDS),
        operational_lookback_quarters=selected,
        original_reference_lookback_quarters=ORIGINAL_LOOKBACK,
        all_training_quarters="2006Q1-2019Q4",
        trained_supervised_origins="2011Q1-2019Q4",
        comparison="Dev 2020Q1-2022Q4",
        dev_previously_used_for_stage04_model_selection=True,
        independent_test_claim=False,
        conditional_on_fixed_asof_9d_context=True,
        constant_temporal_core_capacity=CORE_LOOKBACK,
        per_seed_errors_averaged_not_seed_ensemble_predictions=True,
        development_minimum_mean_mae_length=selected,
        selected_length_is_dev_conditional=True,
        trained_length_seed_jobs=len(LENGTHS)*len(SEEDS),
        train_internal_cv=False,
        train_strategy="FULL_TRAIN_SINGLE_FIT_PER_SEED_AND_LENGTH",
        dev_selected_lookback_quarters=selected,
        reference_lookback_quarters=ORIGINAL_LOOKBACK,
        dev_policy_selection=selection,
        test_opened=False)
    json_write(root/"main_experiment_report.json",report)
    json_write(root/"run_manifest.json",dict(
        status="COMPLETE_MAIN_DEV_ANALYSIS",
        signature=digest,contract=contract,
        completed_runs=200,test_opened=False,
        training_source_sha256=contract["source_sha256"],
        analysis_source_sha256=sha(SOURCE),
        finished_utc=datetime.now(timezone.utc).isoformat()))
    print("05_MAIN_DEV_ANALYSIS_COMPLETE",
          "TRIALS",200,"ORIGINAL_REFERENCE",ORIGINAL_LOOKBACK,
          "SELECTED_DEV_LENGTH",selected,flush=True)
    return report


def analyze_frozen_experiment():
    """Regenerate Dev summaries, policy selection and four plots: no new fit.

    The training-source fingerprint is kept verbatim from its original
    manifest even if reporting code subsequently changes. A separate
    analysis-source SHA documents the current descriptive analysis code.
    Do not rewrite original job signatures or train another model.
    """
    m,panel,lock,upstream_manifest=upstream()
    path=RESULTS/"run_manifest.json"
    if not path.is_file():
        raise RuntimeError("Main Stage05 requires 200 completed training jobs")
    original=json_read(path)
    contract=original["contract"]
    digest=hashlib.sha256(
        json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if (original["signature"]!=digest
            or original.get("completed_runs")!=len(LENGTHS)*len(SEEDS)
            or original.get("status") not in (
                "COMPLETE_DESCRIPTIVE","COMPLETE_MAIN_DEV_ANALYSIS")
            or contract.get("model")!=MODEL
            or contract.get("formulation")!=MODE
            or tuple(contract.get("seeds",()))!=SEEDS
            or contract.get("lookback_grid")!=list(LENGTHS)
            or contract.get("fixed_epochs")!=75
            or contract.get("train_origin_count")!=36
            or contract.get("fixed_bmtgnn_core_temporal_capacity")!=20
            or contract.get("fixed_model_shape_every_length") is not True
            or contract.get("original_train_dev_sources")!=panel.hashes
            or contract.get("stage04_source_sha256")!=sha(SOURCE04)
            or contract.get("stage04_manifest_sha256")!=
                sha(RESULT04/"run_manifest.json")
            or contract.get("stage04_selection_sha256")!=
                sha(RESULT04/"dev_selection.json")
            or contract.get("test_opened") is not False
            or original.get("test_opened") is not False):
        raise RuntimeError("Frozen Stage05 training/data contract mismatch")
    report=finalize(RESULTS,contract,digest,panel)
    # The training contract/source SHA from the original 200 fits is kept
    # distinct from the analysis source SHA added in finalize().
    assert json_read(path)["contract"]["source_sha256"]==contract["source_sha256"]
    return report


def run(args):
    if args.status:
        state=(json_read(RESULTS/"run_manifest.json")
               if (RESULTS/"run_manifest.json").is_file() else None)
        print("05_STATUS",
              (state or {}).get("status","NOT_STARTED"),
              "current_jobs",len(list((RESULTS/"trials").glob("*.json"))),
              "CANONICAL_RESULTS",RESULTS,
              "TRAIN_INTERNAL_VALIDATION",False)
        return
    if args.dry_run:
        print("05_MAIN_EXPERIMENT_BMTGNN_ONLY_FIXED_CAPACITY",
              "TRAIN 2006–2019 FULL",
              "TRAIN TARGET ORIGINS 2011–2019 (36 COMMON)",
              "EVAL DEV 2020–2022 (12)",
              "FIXED_EPOCHS 75","L=1..20",
              "BMTGNN_STRUCTURAL",
              "SEEDS 2026–2035 (10)",
              "TOTAL TRIALS 200",
              "CORE_CAPACITY 20 FOR EVERY L",
              "ASOF_9D_CONTEXT FIXED",
              "ORIGINAL L12 NOT RETRAINED FROM STAGE04",
              "DEV_SELECT_MIN_10_SEED_MEAN_3AXIS_MAE",
              "ORIGINAL_L12_REFERENCE_ONLY",
              "FOUR_DESCRIPTIVE_PLOTS",
              "NO 2018–2019 HOLDOUT","NO TEST",flush=True)
        return
    if args.analyze:
        analyze_frozen_experiment()
        return
    if not args.run:
        raise ValueError("Use --dry-run, --status or --run")
    m,panel,lock,manifest=upstream()
    contract,digest=run_contract(m,panel,lock,manifest)
    dest=RESULTS/"run_manifest.json"
    previous=json_read(dest) if dest.is_file() else None
    if previous and previous.get("completed_runs")==200:
        raise RuntimeError(
            "The 200 original Stage05 jobs are already complete; "
            "use --analyze to refresh Dev selection and four plots, "
            "never retrain or relabel their source provenance")
    if previous and previous.get("signature")!=digest:
        raise RuntimeError("New Stage05 run is inconsistent with saved data")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    RESULTS.mkdir(parents=True,exist_ok=True)
    json_write(dest,dict(
        status="RUNNING",signature=digest,contract=contract,
        train_internal_validation=False,test_opened=False))
    torch.set_num_threads(4)
    inp=Inputs(panel)
    device=torch.device(args.device)
    for L in LENGTHS:
        for seed in SEEDS:
            stem=f"L{L:02d}__{MODEL}__s{seed}"
            meta=RESULTS/"trials"/(stem+".json")
            data=RESULTS/"trials"/(stem+".npz")
            if meta.is_file() and data.is_file():
                info=json_read(meta)
                if (info["signature"]!=digest or
                        info["prediction_sha256"]!=sha(data)):
                    raise RuntimeError("Stage05 cached result contract changed")
                continue
            if meta.exists() or data.exists():
                raise RuntimeError("Incomplete Stage05 job")
            prediction=train_one(inp,MODEL,seed,L,device)
            save_npz(data,prediction=prediction)
            json_write(meta,dict(
                signature=digest,L=L,model=MODEL,seed=seed,
                fixed_epochs=m.STAGE04_ACTIVE_EPOCHS,
                prediction_sha256=sha(data),
                trained_2006_2019_only=True,
                train_internal_validation=False,test_opened=False))
            print("05_TRAIN_DEV",stem,flush=True)
    return finalize(RESULTS,contract,digest,panel)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--status",action="store_true")
    p.add_argument("--analyze",action="store_true")
    p.add_argument("--run",action="store_true")
    p.add_argument("--device",choices=("cpu","cuda"),
                   default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__=="__main__":
    main()
