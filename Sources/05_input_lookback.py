#!/usr/bin/env python3
"""PRISM Supplementary S05: input-history-length SENSITIVITY, not optimization.

The deployed forecasting method remains Stage04 exactly: 12 input quarters,
five selected GNNs, five matched seeds, equal-weight Structural ensemble,
Standard/raw-zero/missing-mask preprocessing, Train 2006-19 fitted once,
and frozen weights with freshly available past X for each Dev forecast.

S05 is an optional, DESCRIPTIVE robustness analysis of L=1..20. Lengths
share the same Train-only supervised origin sets:
  2011-15 (20 origins): epoch search fit; 2016-17: epoch assessment
  2011-17 (28 origins): lookback-sensitivity fit
  2018-19 (8 origins): lookback-sensitivity measurements
40-L >= L explains the study's 20-quarter upper support limit. The 1:1
sample-support rule is an explicit design choice, not a statistical law.
The observed smallest error is recorded honestly as a descriptive fact;
it does NOT change the original 12-quarter forecasting method.

Previously completed 500 training trials are cached at
Results/05_input_lookback/trials/ (500 JSON + 500 NPZ). --rebuild reads
and validates these 500 trials, regenerates only the supplementary tables
and figures, and DELETES superseded optimization/final-model artifacts.
It NEVER fits or reruns a neural network, never archives the superseded
11-quarter optimization, and never changes any trial predictions.

Use --run only when training trials are genuinely absent/incomplete.
2023-25 Test is never opened. Historic economic inputs are Stage03's
latest-revised data; we do not claim real-time publication-vintage replay.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"Sources/05_input_lookback.py"
STAGE04_SOURCE=ROOT/"Sources/04_gnn_benchmark.py"
STAGE04_RESULTS=ROOT/"Results/04_gnn_benchmark"
RESULTS=ROOT/"Results/05_input_lookback"
ORIGINAL_LOOKBACK=12
EARLY_END=40           # 2006Q1–2015Q4
EPOCH_END=48           # 2016Q1–2017Q4
TRAIN_END=56           # 2018Q1–2019Q4
DEV_END=68             # 2020Q1–2022Q4; not used for sensitivity
FIRST_CONTEXT_ORIGIN=12
AXES=("profitability","productivity","growth")
OLD_TRAINING_FINGERPRINT="abb0e4fad9317d6291c577362cc5483a2273a83f9eb20f35a9a354768441d50b"
OLD_TRAINING_SOURCE="2a6d884ad88476afd12f10a86e27bfe6ae6bd296367bb89c06be0c2b4a69af83"


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path:Path,record:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".writing")
    tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False,
                              allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def atomic_npz(path:Path,**arrays)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".writing.npz")
    np.savez_compressed(tmp,**arrays)
    os.replace(tmp,path)


def read_json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def stage04():
    spec=importlib.util.spec_from_file_location("_prism05_supplementary04",
                                                 STAGE04_SOURCE)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Ensemble:
    models:tuple[str,...]
    seeds:tuple[int,...]
    weights:tuple[float,...]
    mode:str
    lock_sha256:str
    manifest_sha256:str


def original_04()->Ensemble:
    m=stage04()
    lock_file=STAGE04_RESULTS/"selected_ensemble_pool.json"
    manifest_file=STAGE04_RESULTS/"run_manifest.json"
    report_file=STAGE04_RESULTS/"selected_mode_and_ensemble.json"
    if not all(f.is_file() for f in (lock_file,manifest_file,report_file)):
        raise RuntimeError("Supplementary requires completed Stage04 ensemble")
    lock,manifest,report=map(read_json,(lock_file,manifest_file,report_file))
    if (manifest.get("status")!="PASS"
            or manifest.get("planned_runs")!=762
            or manifest.get("completed_runs")!=762
            or manifest.get("signature",{}).get("source_sha256")!=sha256(STAGE04_SOURCE)
            or report.get("status")!="COMPLETE_ALL_THREE_MODES"
            or lock.get("status")!="LOCKED"
            or lock.get("model_family")!="GNN_ONLY"
            or lock.get("prediction_mode")!=report.get("selected_mode")
            or lock.get("models")!=report.get("selected_ensemble_models")
            or lock.get("test_opened") is not False):
        raise RuntimeError("Stage04 is not the current completed model family")
    names=tuple(lock["models"])
    seeds=tuple(lock["seeds"])
    weights=tuple(float(x) for x in lock["equal_weights"])
    if (len(names)!=5 or len(set(names))!=5
            or not set(names).issubset(set(m.GNN_MODELS))
            or seeds!=tuple(m.DEFAULT_SEEDS)
            or len(weights)!=5 or not np.allclose(
                weights,.2,atol=1e-12,rtol=0)
            or tuple(manifest["signature"][
                "economic_prior_train_coefficient"])!=tuple(
                    m.STAGE04_FIXED_ALPHA)):
        raise RuntimeError("Stage04 fixed ensemble/Train prior changed")
    for file,field in (
        ("raw_train.csv","train_sha256"),("raw_val.csv","dev_sha256")):
        expected=manifest["signature"][field]
        if (sha256(ROOT/"Data/03_integrated_dataset"/file)!=expected
                or lock["input_sha256"][
                    "Data/03_integrated_dataset/"+file]!=expected):
            raise RuntimeError("Stage04 selected input data SHA mismatch")
    for model in names:
        if (m.fixed_preprocessing_policy(model)!=("standard","mask")
                or lock["input_policies"][model]!=dict(
                    scaler="standard",missing="mask")):
            raise RuntimeError("Fixed Stage04 neural preprocessing changed")
    return Ensemble(names,seeds,weights,lock["prediction_mode"],
                    sha256(lock_file),sha256(manifest_file))


def study_design():
    """One fixed quantitative support rule, not a performance-tuned ceiling."""
    upper=EARLY_END//2  # early supervised origins: 40-L >= L
    lengths=tuple(range(1,upper+1))
    early=np.arange(upper,EARLY_END,dtype=int)
    epoch=np.arange(EARLY_END,EPOCH_END,dtype=int)
    refit=np.arange(upper,EPOCH_END,dtype=int)
    measure=np.arange(EPOCH_END,TRAIN_END,dtype=int)
    assert (len(lengths)==20 and ORIGINAL_LOOKBACK in lengths
            and len(early)==20 and len(refit)==28
            and len(epoch)==8 and len(measure)==8)
    return dict(lengths=lengths,upper=upper,
                early=early,epoch=epoch,refit=refit,measure=measure)


def three_axis_metrics(pred,truth,previous):
    pred,truth,previous=(np.asarray(a,dtype=np.float64)
                         for a in (pred,truth,previous))
    if (pred.shape!=truth.shape or pred.shape!=previous.shape
            or pred.ndim!=3 or pred.shape[1:]!=(59,3)
            or not all(np.isfinite(a).all() for a in (
                pred,truth,previous))):
        raise ValueError("Unexpected three-axis historical prediction")
    errors=pred-truth
    return dict(
        axis_mae=float(np.abs(errors).mean()),
        ici_mae=float(np.abs(errors.mean(axis=-1)).mean()),
        profitability_mae=float(np.abs(errors[...,0]).mean()),
        productivity_mae=float(np.abs(errors[...,1]).mean()),
        growth_mae=float(np.abs(errors[...,2]).mean()),
        persistence_mae=float(np.abs(previous-truth).mean()))


class Inputs:
    """Stage04's unchanged as-of context, economic prior, 59×157 inputs."""
    def __init__(self,panel):
        self.p=panel
        self.m=stage04()
        n,d,info=self.m.economic_components(panel)
        self.prior=np.zeros_like(panel.y)
        idx=np.arange(FIRST_CONTEXT_ORIGIN,len(panel.y))
        self.prior[idx]=self.m.component_structural_prior(
            n,d,info,idx,np.asarray(self.m.STAGE04_FIXED_ALPHA,float))
        self.graph_cache={}

    def graph(self,fit_end:int):
        if fit_end not in (EARLY_END,EPOCH_END):
            raise ValueError("Graph must be from an earlier Train fold")
        if fit_end not in self.graph_cache:
            p=self.p
            train=self.m.Panel(p.x,p.y,p.quarters,p.nodes,p.names,
                               p.features,fit_end,p.hashes)
            self.graph_cache[fit_end]=self.m.train_graph(train)
        return self.graph_cache[fit_end]

    def fold(self,L,fit_end,train_origins,holdout_origins):
        p=self.p
        t=np.asarray(train_origins,dtype=int)
        v=np.asarray(holdout_origins,dtype=int)
        if (L not in study_design()["lengths"]
                or fit_end not in (EARLY_END,EPOCH_END)
                or len(t)==0 or len(v)==0
                or t.min()<max(L,FIRST_CONTEXT_ORIGIN)
                or t.max()>=fit_end or v.min()<fit_end
                or v.max()>=TRAIN_END):
            raise ValueError("Leaky/unmatched lookback sensitivity fold")
        c,s=self.m.fit_normalizer(p.x[:fit_end],"standard")
        present=np.isfinite(p.x)
        x=((np.where(present,p.x,0.)-c)/s).astype(np.float32)
        mask=(~present).astype(np.float32)
        yc=p.y[:fit_end].mean((0,1))
        ys=p.y[:fit_end].std((0,1))
        ys=np.where(ys>1e-8,ys,1.)
        normalized_y=((p.y-yc)/ys).astype(np.float32)
        normalized_prior=((self.prior-yc)/ys).astype(np.float32)
        def pack(ids,prefix):
            return {
                prefix+"_x":np.stack([x[o-L:o] for o in ids]),
                prefix+"_mask":np.stack([mask[o-L:o] for o in ids]),
                prefix+"_y":np.stack([normalized_y[o:o+1] for o in ids]),
                prefix+"_last":((self.m.past_scores(p,ids)-yc)/ys).astype(np.float32),
                prefix+"_context":self.m.known_score_context(p,ids,yc,ys),
                prefix+"_structure":normalized_prior[ids].astype(np.float32)}
        out=dict(
            **pack(t,"train"),**pack(v,"val"),
            x_center=c.astype(np.float32),x_scale=s.astype(np.float32),
            y_center=yc.astype(np.float32),y_scale=ys.astype(np.float32))
        if (out["train_x"].shape!=(len(t),L,59,157)
                or out["val_y"].shape!=(len(v),1,59,3)
                or any(not np.isfinite(a).all() for a in out.values())):
            raise RuntimeError("Stage05 input shape/nonfinite value")
        return out


def train_network(inputs,model_name,seed,length,mode,fold,fit_end,device,
                  epochs=None):
    """Original Stage04 architecture/loss/optimizer/Bayesian inference."""
    m=stage04()
    m.seed_all(seed)
    network=m.ThreeAxisStructureGNN(
        model_name,fold["x_center"],fold["x_scale"],
        fold["y_center"],fold["y_scale"],
        num_features=157,nodes=59,prediction_mode=mode)
    if length!=m.LOOKBACK:
        network.base=m.ForecastModel(
            model_name,157,59,m.STAGE04_ACTIVE_WIDTH,length,m.HORIZON)
    network=network.to(device)
    optimizer=torch.optim.AdamW(
        network.parameters(),lr=m.FIXED_LEARNING_RATE,weight_decay=1e-4)
    keys=("train_x","train_mask","train_y","train_last",
          "train_structure","train_context",
          "val_x","val_mask","val_y","val_last",
          "val_structure","val_context")
    tensors={key:torch.as_tensor(fold[key],device=device) for key in keys}
    graph=torch.as_tensor(inputs.graph(fit_end),device=device)
    scale=torch.as_tensor(fold["y_scale"],device=device)
    iterations=m.STAGE04_ACTIVE_EPOCHS if epochs is None else int(epochs)
    best,best_epoch,stale=float("inf"),1,0
    for epoch in range(1,iterations+1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        pred=network(tensors["train_x"],tensors["train_mask"],graph,
                     tensors["train_last"],tensors["train_structure"],
                     tensors["train_context"])
        loss=m.three_axis_weighted_huber(
            pred,tensors["train_y"],tensors["train_last"],scale)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite weighted Stage04 loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),5.,
                                       error_if_nonfinite=True)
        optimizer.step()
        if epochs is None:
            if model_name=="B-MTGNN":
                network.train()
                draws=m.STAGE04_BMTGNN_INNER_MC_SAMPLES
            else:
                network.eval()
                draws=1
            with torch.inference_mode():
                p=torch.stack([
                    network(tensors["val_x"],tensors["val_mask"],graph,
                            tensors["val_last"],tensors["val_structure"],
                            tensors["val_context"])
                    for _ in range(draws)]).mean(0)
                score=float(((p-tensors["val_y"]).abs()*scale).mean())
            if score<best-1e-6:
                best,best_epoch,stale=score,epoch,0
            else:
                stale+=1
            if stale>=m.STAGE04_ACTIVE_PATIENCE:
                break
    if epochs is None:
        return dict(best_epoch=int(best_epoch),epoch_mae=float(best))
    result=m.mtgnn_bayesian_inference(
        network,tensors,graph,model_name,seed,
        fold["y_center"],fold["y_scale"])
    return dict(prediction=result["prediction"])


def train_one(inputs,ensemble,L,model,seed,design,device):
    early=inputs.fold(L,EARLY_END,design["early"],design["epoch"])
    epoch=train_network(
        inputs,model,seed,L,ensemble.mode,early,EARLY_END,device)
    second=inputs.fold(L,EPOCH_END,design["refit"],design["measure"])
    forecast=train_network(
        inputs,model,seed,L,ensemble.mode,second,EPOCH_END,device,
        epochs=epoch["best_epoch"])["prediction"]
    truth=inputs.p.y[design["measure"]]
    past=stage04().past_scores(inputs.p,design["measure"])
    return dict(**epoch,**three_axis_metrics(forecast,truth,past)),forecast


def trial_paths(root,L,name,seed):
    stem=f"L{L:02d}__{name}__s{seed}"
    return root/"trials"/(stem+".json"),root/"trials"/(stem+".npz")


def read_trial(root,fingerprint,L,name,seed):
    meta,array=trial_paths(root,L,name,seed)
    if meta.is_file()!=array.is_file():
        raise RuntimeError("Incomplete Stage05 training trial; do not reuse")
    if not meta.is_file():
        return None
    record=read_json(meta)
    if (record.get("contract_sha256")!=fingerprint
            or record.get("length")!=L or record.get("model")!=name
            or record.get("seed")!=seed):
        raise RuntimeError(f"Stale Stage05 training record: {meta.name}")
    with np.load(array,allow_pickle=False) as z:
        if z.files!=["prediction"]:
            raise RuntimeError("Unexpected trained trial prediction schema")
        pred=z["prediction"].copy()
    if pred.shape!=(8,59,3) or not np.isfinite(pred).all():
        raise RuntimeError(f"Invalid trained trial array {array.name}")
    return record,pred


def write_trial(root,fingerprint,L,name,seed,details,pred):
    meta,array=trial_paths(root,L,name,seed)
    if meta.exists() or array.exists():
        raise RuntimeError("Never overwrite existing training results")
    atomic_npz(array,prediction=pred)
    atomic_json(meta,dict(
        contract_sha256=fingerprint,length=L,model=name,
        seed=seed,**details))


def digest_trial_files(root)->str:
    h=hashlib.sha256()
    files=sorted((root/"trials").iterdir())
    for file in files:
        if not file.is_file():
            raise RuntimeError("Unexpected nested trial file")
        h.update(file.name.encode("utf-8")+b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b""):
                h.update(block)
    return h.hexdigest()


def training_fingerprint(root,ensemble,panel,design):
    """Trust original run only if its data/model training contract matches.

    The historical hash is retained ON THE 500 MODEL TRAINING FILES ONLY;
    no former optimization status, selected length or output survives.
    """
    manifest=root/"run_manifest.json"
    if not manifest.is_file():
        if (root/"trials").exists():
            raise RuntimeError("Training outputs without provenance manifest")
        return None
    state=read_json(manifest)
    if state.get("status")=="PASS_SUPPLEMENTARY":
        if (state.get("stage04_sha256")!=sha256(STAGE04_SOURCE)
                or state.get("stage04_lock_sha256")!=ensemble.lock_sha256
                or state.get("model_count")!=len(ensemble.models)
                or state.get("seed_count")!=len(ensemble.seeds)):
            raise RuntimeError("Supplementary report uses different Stage04")
        return state["training_fingerprint"]
    old=state.get("contract",{})
    if (state.get("status")=="PASS"
            and state.get("contract_sha256")==OLD_TRAINING_FINGERPRINT
            and old.get("source_sha256")==OLD_TRAINING_SOURCE
            and old.get("source04_sha256")==sha256(STAGE04_SOURCE)
            and old.get("selected04",{}).get("lock_hash")==ensemble.lock_sha256
            and old.get("selected04",{}).get("manifest_hash")==ensemble.manifest_sha256
            and old.get("panel_sha256")==panel.hashes
            and old.get("lookbacks")==list(design["lengths"])
            and old.get("nodes")==59 and old.get("features")==157
            and old.get("epoch_cap")==stage04().STAGE04_ACTIVE_EPOCHS
            and old.get("equal_common_early_fit_origins")==design["early"].tolist()
            and old.get("equal_common_preselection_fit_origins")==design["refit"].tolist()
            and old.get("train_only_epoch_origins")==design["epoch"].tolist()
            and old.get("train_only_lookback_choice_origins")==design["measure"].tolist()):
        return OLD_TRAINING_FINGERPRINT
    if (state.get("status")=="TRAINING_SUPPLEMENTARY"
            and state.get("stage04_sha256")==sha256(STAGE04_SOURCE)
            and state.get("source_sha256")==sha256(SOURCE)):
        return state["training_fingerprint"]
    raise RuntimeError("Unrecognized Stage05 trial provenance; no cache mixing")


def current_fingerprint(ensemble,panel,design):
    inputs=dict(
        source_sha256=sha256(SOURCE),
        stage04_sha256=sha256(STAGE04_SOURCE),
        models=list(ensemble.models),seeds=list(ensemble.seeds),
        weights=list(ensemble.weights),mode=ensemble.mode,
        stage04_lock_sha256=ensemble.lock_sha256,
        data_sha256=panel.hashes,
        historical_fit_origins=design["early"].tolist(),
        epoch_origins=design["epoch"].tolist(),
        sensitivity_fit_origins=design["refit"].tolist(),
        sensitivity_measure_origins=design["measure"].tolist())
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def score_500_trials(root,fingerprint,ensemble,panel,design):
    """Recompute the complete 20-length curve from unmodified TRAINING NPZs."""
    expected=(
        len(design["lengths"])*len(ensemble.models)*len(ensemble.seeds))
    files=list((root/"trials").iterdir())
    if (len(files)!=2*expected
            or sum(f.suffix==".json" for f in files)!=expected
            or sum(f.suffix==".npz" for f in files)!=expected):
        raise RuntimeError("Incomplete 500-run training cache")
    previous=stage04().past_scores(panel,design["measure"])
    truth=panel.y[design["measure"]]
    records=[]
    yearly=[]
    for L in design["lengths"]:
        network_preds=[]
        for name in ensemble.models:
            seed_predictions=[]
            for seed in ensemble.seeds:
                result=read_trial(root,fingerprint,L,name,seed)
                if result is None:
                    raise RuntimeError(f"Missing trained trial: {L}/{name}/{seed}")
                row,forecast=result
                scored=three_axis_metrics(forecast,truth,previous)
                if any(not np.isclose(row[key],scored[key],rtol=0,atol=1e-10)
                       for key in scored):
                    raise RuntimeError(f"Stored training score mismatch {L}/{name}/{seed}")
                seed_predictions.append(forecast)
            network_preds.append(np.mean(np.stack(seed_predictions),axis=0))
        prediction=np.average(
            np.stack(network_preds),axis=0,weights=ensemble.weights)
        metrics=three_axis_metrics(prediction,truth,previous)
        records.append(dict(
            input_quarters=L,**metrics,
            common_early_fit_origins=len(design["early"]),
            common_sensitivity_fit_origins=len(design["refit"]),
            historical_measure_origins=len(design["measure"]),
            original_stage04_12q_reference=(L==ORIGINAL_LOOKBACK)))
        per_q=np.abs(prediction-truth).mean(axis=(1,2))
        for year,quarter_idx in ((2018,slice(0,4)),(2019,slice(4,8))):
            yearly.append(dict(input_quarters=L,year=year,
                               axis_mae=float(per_q[quarter_idx].mean())))
    table=pd.DataFrame(records).sort_values("input_quarters")
    blocks=pd.DataFrame(yearly)
    reference=float(table.loc[
        table.input_quarters.eq(ORIGINAL_LOOKBACK),"axis_mae"].iloc[0])
    table["mae_minus_original_12q"]=table.axis_mae-reference
    for year in (2018,2019):
        scores=blocks.loc[blocks.year.eq(year)].sort_values("input_quarters")
        ref=float(scores.loc[
            scores.input_quarters.eq(ORIGINAL_LOOKBACK),"axis_mae"].iloc[0])
        table[f"{year}_mae"]=scores.axis_mae.to_numpy()
        table[f"{year}_minus_original_12q"]=scores.axis_mae.to_numpy()-ref
    return table


def plot_four(table,panel,design,output_dir):
    """Every plotted point is from the 500 cached Train-only predictions."""
    output_dir.mkdir(parents=True,exist_ok=True)
    x=table.input_quarters.to_numpy()
    figures=[]
    def save(fig,name):
        fig.tight_layout()
        target=output_dir/name
        fig.savefig(target,dpi=170,bbox_inches="tight")
        plt.close(fig)
        figures.append(name)
    def reference(ax):
        ax.axvline(ORIGINAL_LOOKBACK,ls="--",color="tab:orange",
                   label="Pre-existing Stage04 12-quarter setting")
    fig,ax=plt.subplots(figsize=(11.4,5.0))
    ax.plot(x,table.axis_mae,"o-",lw=1.3,markersize=4,
            label="2018-2019 historical sensitivity MAE")
    reference(ax)
    ax.set(title="Supplementary: input-history sensitivity (no length selection)",
           xlabel="Input history (quarters)",ylabel="3-axis MAE, lower is better")
    ax.set_xticks(x)
    ax.grid(alpha=.3);ax.legend()
    save(fig,"S05_01_input_length_mae.png")

    fig,ax=plt.subplots(figsize=(11.4,5.4))
    for metric,title in (
        ("profitability_mae","Profitability"),
        ("productivity_mae","Productivity"),
        ("growth_mae","Growth"),("ici_mae","ICI")):
        ax.plot(x,table[metric],"o-",ms=3,label=title)
    reference(ax)
    ax.set(title="Sensitivity of three original axes and ICI",
           xlabel="Input history (quarters)",ylabel="Mean absolute error")
    ax.set_xticks(x);ax.grid(alpha=.3);ax.legend()
    save(fig,"S05_02_axes_and_ici.png")

    fig,ax=plt.subplots(figsize=(11.4,4.8))
    ax.axhline(0,color="black",lw=1,alpha=.6)
    ax.bar(x,table.mae_minus_original_12q)
    reference(ax)
    ax.set(title="Descriptive differences relative to fixed Stage04 12 quarters",
           xlabel="Input history (quarters)",
           ylabel="Historical MAE minus 12Q MAE (negative = lower)")
    ax.set_xticks(x);ax.grid(axis="y",alpha=.25);ax.legend()
    save(fig,"S05_03_difference_from_12q.png")

    fig,axes=plt.subplots(1,2,figsize=(14.2,5.0))
    for year in (2018,2019):
        axes[0].plot(x,table[f"{year}_minus_original_12q"],
                     "o-",ms=3,label=str(year))
    axes[0].axhline(0,color="black",ls=":",alpha=.7)
    reference(axes[0])
    axes[0].set(title="Only two historical annual comparison blocks",
                xlabel="Input history (quarters)",
                ylabel="Yearly MAE difference from original 12Q")
    axes[0].grid(alpha=.25);axes[0].legend(fontsize=8)
    axes[1].plot(x,EARLY_END-x,label="Feasible early-fit origins")
    axes[1].plot(x,x,"--",label="One fit origin per input quarter")
    axes[1].axhline(len(design["early"]),color="tab:green",
                    label="Same 20 origins actually used")
    reference(axes[1])
    axes[1].set(title="A declared sample-support boundary, not an optimum",
                xlabel="Input history (quarters)",ylabel="Quarterly origin count")
    axes[1].grid(alpha=.25);axes[1].legend(fontsize=8)
    save(fig,"S05_04_years_and_data_support.png")
    return figures


def rebuild(root,ensemble,panel,design):
    """Delete ALL former 05 non-training artifacts. Never retrain/rename L11."""
    if not root.is_dir() or not (root/"trials").is_dir():
        raise RuntimeError("No trained 500-run sensitivity cache to report")
    fingerprint=training_fingerprint(root,ensemble,panel,design)
    if fingerprint is None:
        raise RuntimeError("Missing provenance for cached model training")
    before=digest_trial_files(root)
    table=score_500_trials(root,fingerprint,ensemble,panel,design)
    if len(table)!=20:
        raise RuntimeError("Expected exactly 20 descriptive lookback rows")
    minimum=table.sort_values(["axis_mae","input_quarters"]).iloc[0]
    reference=table.loc[
        table.input_quarters.eq(ORIGINAL_LOOKBACK)].iloc[0]
    with tempfile.TemporaryDirectory(
            prefix="prism05_supplementary_build_",dir=ROOT/"Results") as tmp:
        stage=Path(tmp)
        table.to_csv(stage/"supplementary_sensitivity.csv",
                     index=False,encoding="utf-8-sig")
        figures=plot_four(table,panel,design,stage/"plots")
        report=dict(
            stage="S05",status="PASS_SUPPLEMENTARY",
            purpose="DESCRIPTIVE_INPUT_LENGTH_SENSITIVITY_ONLY",
            operational_lookback=ORIGINAL_LOOKBACK,
            operational_setting_source="Unchanged pre-existing Stage04",
            no_new_operational_model=True,
            GNNs=list(ensemble.models),matched_seeds=list(ensemble.seeds),
            weights=list(ensemble.weights),forecast_mode=ensemble.mode,
            tested_input_lengths=list(design["lengths"]),
            evaluation_quarters=panel.quarters[EPOCH_END:TRAIN_END],
            common_early_fit_origins=len(design["early"]),
            common_sensitivity_fit_origins=len(design["refit"]),
            observation_years=[2018,2019],
            original_12q_historical_mae=float(reference.axis_mae),
            descriptive_smallest_mae_length=int(minimum.input_quarters),
            descriptive_smallest_mae=float(minimum.axis_mae),
            descriptive_minimum_minus_12q=float(
                minimum.axis_mae-reference.axis_mae),
            analysis_not_a_model_policy=True,
            statistical_equivalence_not_claimed=True,
            global_optimum_not_claimed=True,
            prior_calibrated_on_all_2006_2019_train=True,
            stage04_architectures_chosen_using_dev=True,
            frozen_forecasting_policy_unchanged=True,
            model_training_executed_during_rebuild=False,
            ici_definition="Arithmetic mean of original three axes",
            test_opened=False,
            plots=["plots/"+p for p in figures])
        atomic_json(stage/"supplementary_report.json",report)
        digest_after=digest_trial_files(root)
        if digest_after!=before:
            raise RuntimeError("Model training cache was modified during reporting")
        state=dict(
            stage="S05",status="PASS_SUPPLEMENTARY",
            role="SUPPLEMENTARY_DESCRIPTIVE_ONLY",
            adopted_lookback_quarters=ORIGINAL_LOOKBACK,
            trial_count=500,model_count=len(ensemble.models),
            seed_count=len(ensemble.seeds),
            stage04_sha256=sha256(STAGE04_SOURCE),
            stage04_lock_sha256=ensemble.lock_sha256,
            source_sha256=sha256(SOURCE),
            training_fingerprint=fingerprint,
            training_trials_sha256=before,
            neural_networks_trained_during_rebuild=0,
            test_opened=False,
            figures=report["plots"],
            generated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(stage/"run_manifest.json",state)
        # The user explicitly requested NO archive, no selected-L11 legacy
        # results and no remaining optimization result paths. Keep only
        # the 500 trained trial records and their prediction NPZs.
        removed=[]
        for item in root.iterdir():
            if item.name=="trials":
                continue
            removed.append(item.name)
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for item in stage.iterdir():
            shutil.move(str(item),str(root/item.name))
        if digest_trial_files(root)!=before:
            raise RuntimeError("500 trained model outputs changed during cleanup")
        allowed={"trials","plots","supplementary_sensitivity.csv",
                 "supplementary_report.json","run_manifest.json"}
        if {p.name for p in root.iterdir()}!=allowed:
            raise RuntimeError("Superseded Stage05 files survived cleanup")
        print("SUPPLEMENTARY_05_PASS",
              "TRIALS",500,"FIXED_L",ORIGINAL_LOOKBACK,
              "DESCRIPTIVE_LOWEST_L",int(minimum.input_quarters),
              "PLOTS",len(figures),"RETRAINED",0,flush=True)
        print("REMOVED_OBSOLETE_STAGE05_ARTIFACTS",removed,flush=True)
    return report


def run_missing_trials(root,ensemble,panel,design,device,budget_minutes):
    """Optional fresh/resume path only; no final model / length optimization."""
    root.mkdir(parents=True,exist_ok=True)
    info=root/"run_manifest.json"
    previous=read_json(info) if info.is_file() else None
    if previous and previous.get("status")=="PASS_SUPPLEMENTARY":
        return rebuild(root,ensemble,panel,design)
    if previous:
        fingerprint=training_fingerprint(root,ensemble,panel,design)
    else:
        fingerprint=current_fingerprint(ensemble,panel,design)
    if fingerprint is None:
        fingerprint=current_fingerprint(ensemble,panel,design)
    begun=time.perf_counter()
    state=dict(stage="S05",status="TRAINING_SUPPLEMENTARY",
               source_sha256=sha256(SOURCE),
               stage04_sha256=sha256(STAGE04_SOURCE),
               training_fingerprint=fingerprint,
               model_count=len(ensemble.models),seed_count=len(ensemble.seeds),
               adopted_lookback_quarters=ORIGINAL_LOOKBACK,
               test_opened=False)
    atomic_json(info,state)
    inputs=Inputs(panel)
    completed=0
    for L in design["lengths"]:
        for name in ensemble.models:
            for seed in ensemble.seeds:
                if read_trial(root,fingerprint,L,name,seed) is not None:
                    completed+=1
                    continue
                if time.perf_counter()-begun>budget_minutes*60:
                    state.update(status="TRAINING_SUPPLEMENTARY",
                                 finished_trials=completed,
                                 next_job=f"L{L}:{name}:{seed}")
                    atomic_json(info,state)
                    print("SUPPLEMENTARY_05_RESUMABLE",completed,"/500")
                    return state
                scores,pred=train_one(
                    inputs,ensemble,L,name,seed,design,device)
                write_trial(root,fingerprint,L,name,seed,scores,pred)
                completed+=1
                print("SUPPLEMENTARY_05_TRIAL",completed,"/500",
                      "L",L,name,seed,flush=True)
    return rebuild(root,ensemble,panel,design)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild",action="store_true",
                        help="Rebuild supplementary from saved 500 trials; ZERO model fits")
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--run",action="store_true",
                        help="Fit any missing trials; NOT needed for completed 500-trial cache")
    parser.add_argument("--device",choices=("cpu","cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-minutes",type=float,default=25.)
    args=parser.parse_args()
    ensemble=original_04()
    panel=stage04().load_panel(ROOT)
    if (panel.x.shape!=(68,59,157) or panel.y.shape!=(68,59,3)
            or panel.train_count!=56):
        raise RuntimeError("Original 59×157 Stage03 Train/Dev panel changed")
    design=study_design()
    if args.dry_run:
        print("S05_SUPPLEMENTARY_ONLY L=1..20 / original 04 L12 frozen")
        print("TRAIN EPOCHS 2016-2017; SENSITIVITY 2018-2019; NO MODEL CHOICE")
        print("MODELS",ensemble.models,"SEEDS",ensemble.seeds,"TRIALS",500)
        return
    if args.status:
        manifest=RESULTS/"run_manifest.json"
        previous=read_json(manifest) if manifest.is_file() else {}
        print("S05_STATUS",previous.get("status","READY"),
              "TRIAL_FILES",len(list((RESULTS/"trials").glob("*.npz"))),
              "OPERATIONAL_LOOKBACK",ORIGINAL_LOOKBACK)
        return
    if args.run:
        if args.max_minutes<=0:
            parser.error("--max-minutes must be positive")
        device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        run_missing_trials(RESULTS,ensemble,panel,design,device,args.max_minutes)
        return
    if args.rebuild:
        rebuild(RESULTS,ensemble,panel,design)
        return
    parser.error("Use --rebuild for existing results, --status, --dry-run or --run")


if __name__=="__main__":
    main()
