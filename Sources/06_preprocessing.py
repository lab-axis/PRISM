#!/usr/bin/env python3
"""PRISM 06: sequential preprocessing optimization, no complete grid.

Use Stage04's FINAL fixed 12-quarter Structural 5-GNN × 5-seed ensemble,
the same 2006Q1-2019Q4 FULL Train and frozen 2020Q1-2022Q4 Dev
next-quarter forecasts. Stage04's Train-only epoch count for each GNN
and seed is LOCKED and reused. There is NO Train-internal evaluation
and NO repeated fitting on Dev labels; Dev only scores policies.

Three forward sequential factors (each incumbent + five alternatives):
  missing representations -> scaling -> extreme-value processing.
Then reverse each adopted component once as a conditional ablation.
Each stage adopts a change only when overall 3-axis MAE improves, ICI
MAE does not worsen, and at least two of three individual Dev years
improve against that stage's incumbent. Otherwise retain incumbent.

Strict distribution safety: every fitted scaler, quantile, median, MAD,
maxabs, clipping bound and fallback comes from observed 2006-2019 Train
cells ONLY; no Dev/test fit, ever. Stateful causal imputations for
missing raw economic sources read only earlier rows of that same
industry/feature. All missing masks remain from ORIGINAL missingness.
First 42 derived Target-history channels never get statistical fills,
and outlier transforms touch only 115 source features. Structural prior
uses exactly Stage04's full Train frozen alpha and no Dev targets.

The benchmark's Dev has already been used to select the Stage04 model
family: Stage06 is ADDITIONAL DEVELOPMENT SELECTION, not an independent
out-of-sample generalization claim. The still-closed Test may assess
the final policy in a later, explicitly requested stage.

  py Sources/06_preprocessing.py --dry-run
  py Sources/06_preprocessing.py --status
  py -u Sources/06_preprocessing.py --run --max-minutes 25
The --run command can be resumed without changing any existing models.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"Sources/06_preprocessing.py"
SOURCE04=ROOT/"Sources/04_gnn_benchmark.py"
RESULT04=ROOT/"Results/04_gnn_benchmark"
RESULTS=ROOT/"Results/06_preprocessing"

LOOKBACK=12
TRAIN_END=56
DEV_END=68
SOURCE_FIRST=42
THREE_AXES=("profitability","productivity","growth")
SEEDS=tuple(range(2026,2031))
NUM_MODELS=5
MIN_YEARS_BETTER=2 # 2020,2021,2022; no 4-year 2016-19 internal evaluation
SELECTION_TOL=1e-9
MISSING=("raw_zero","scaled_zero","train_median","train_mean",
         "causal_ffill","causal_seasonal")
NORMALIZATION=("none","standard","robust_iqr","robust_mad",
               "maxabs","quantile_normal")
OUTLIER=("none","winsor005","winsor01","mad_clip5",
         "asinh","signed_log1p")
SLOTS=("missing","normalization","outlier")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".writing")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,
                              allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def write_npz(path:Path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".writing.npz")
    np.savez_compressed(tmp,**arrays)
    os.replace(tmp,path)


@lru_cache(maxsize=1)
def stage04():
    spec=importlib.util.spec_from_file_location("_prism06_stage04",SOURCE04)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Policy:
    missing:str="raw_zero"
    normalization:str="standard"
    outlier:str="none"

    @property
    def id(self)->str:
        return f"m-{self.missing}__n-{self.normalization}__o-{self.outlier}"

    def validate(self):
        if (self.missing not in MISSING
                or self.normalization not in NORMALIZATION
                or self.outlier not in OUTLIER):
            raise ValueError(f"Invalid Stage06 preprocessing {self}")

ORIGINAL=Policy()


def factor_candidates(stage:int,incumbent:Policy)->tuple[Policy,...]:
    """One incumbent and FIVE other values, not full factorial search."""
    incumbent.validate()
    if stage not in (1,2,3):
        raise ValueError(stage)
    component=SLOTS[stage-1]
    options=(MISSING,NORMALIZATION,OUTLIER)[stage-1]
    current=getattr(incumbent,component)
    policies=tuple([incumbent]+[
        replace(incumbent,**{component:variant})
        for variant in options if variant!=current])
    assert len(policies)==6 and len({p.id for p in policies})==6
    return policies


def verify_04_and_load():
    m=stage04()
    files=(RESULT04/"selected_ensemble_pool.json",
           RESULT04/"run_manifest.json",
           RESULT04/"selected_mode_and_ensemble.json")
    if not all(f.is_file() for f in files):
        raise RuntimeError("Stage04 complete fixed ensemble required")
    lock,manifest,report=map(json_read,files)
    models=tuple(lock.get("models",()))
    if (manifest.get("status")!="PASS"
            or manifest.get("completed_runs")!=762
            or manifest.get("signature",{}).get("source_sha256")!=sha(SOURCE04)
            or lock.get("status")!="LOCKED"
            or lock.get("prediction_mode")!="structural"
            or report.get("selected_mode")!="structural"
            or models!=tuple(report.get("selected_ensemble_models",()))
            or len(models)!=NUM_MODELS
            or len(set(models))!=NUM_MODELS
            or not set(models).issubset(set(m.GNN_MODELS))
            or tuple(lock.get("seeds",()))!=SEEDS
            or not np.allclose(lock.get("equal_weights",[]),.2,rtol=0,atol=1e-12)
            or lock.get("test_opened") is not False
            or tuple(manifest["signature"][
                "economic_prior_train_coefficient"])!=tuple(m.STAGE04_FIXED_ALPHA)):
        raise RuntimeError("04 selected model/weights/output/alpha mismatch")
    panel=m.load_panel(ROOT) # Train + Dev ONLY, not test.
    if (panel.x.shape!=(68,59,157)
            or panel.y.shape!=(68,59,3)
            or panel.train_count!=TRAIN_END):
        raise RuntimeError("Required 59-node Stage03 panel changed")
    for basename,key in (
            ("raw_train.csv","train_sha256"),
            ("raw_val.csv","dev_sha256")):
        path=ROOT/"Data/03_integrated_dataset"/basename
        expected=manifest["signature"][key]
        if (sha(path)!=expected
                or lock["input_sha256"][
                    "Data/03_integrated_dataset/"+basename]!=expected):
            raise RuntimeError("Stage03 data not the input used in Stage04")
    if not all(
        m.fixed_preprocessing_policy(name)==("standard","mask")
        for name in models):
        raise RuntimeError("04 baseline policy changed")
    epochs={}
    for name in models:
        for seed in SEEDS:
            key=f"structural__{name}__standard__mask__s{seed}"
            path=RESULT04/"metrics"/(key+".json")
            if not path.is_file():
                raise RuntimeError(f"04 Train-only epoch choice missing: {key}")
            record=json_read(path)
            if (record.get("status")!="PASS"
                    or record.get("model")!=name
                    or record.get("seed")!=seed
                    or record.get("prediction_mode")!="structural"
                    or not 1<=int(record.get("train_epochs",0))<=m.STAGE04_ACTIVE_EPOCHS):
                raise RuntimeError(f"Invalid 04 early-stop epoch record: {key}")
            epochs[(name,seed)]=int(record["train_epochs"])
    return m,panel,models,epochs,lock,manifest


def original_stage04_prediction(panel,models):
    with np.load(RESULT04/"ensemble_predictions.npz",
                 allow_pickle=False) as z:
        pred=z["prediction"].copy()
        truth=z["truth"].copy()
        src=tuple(z["selected_models"].tolist())
    if (pred.shape!=(12,59,3)
            or truth.shape!=pred.shape
            or src!=tuple(models)
            or not np.array_equal(truth,panel.y[TRAIN_END:DEV_END])):
        raise RuntimeError("04 published ensemble does not match Stage03 Dev")
    return pred


def _observed_train_quantile(values,q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        return np.nanquantile(values,q,axis=(0,1))


def observed_train_statistics(raw,fit_end):
    """No computed distribution statistic may use ANY Dev row."""
    if fit_end!=TRAIN_END or raw.shape!=(DEV_END,59,157):
        raise ValueError("Stage06 distribution fit must end at 2019Q4")
    observed=np.asarray(raw[:fit_end],dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(observed,axis=(0,1))
        mean=np.nanmean(observed,axis=(0,1))
        std=np.nanstd(observed,axis=(0,1))
        mad=np.nanmedian(np.abs(observed-median),axis=(0,1))*1.4826
        q=_observed_train_quantile(observed,[.005,.01,.25,.75,.99,.995])
        maxabs=np.nanmax(np.abs(observed),axis=(0,1))
    for values in (median,mean,std,mad,maxabs,q):
        if not np.isfinite(values).all():
            raise RuntimeError("Incomplete 04 Train feature distribution")
    return dict(median=median,mean=mean,std=std,mad=mad,
                maxabs=maxabs,q005=q[0],q01=q[1],
                q25=q[2],q75=q[3],q99=q[4],q995=q[5])


def outlier_transform(raw:np.ndarray,policy:Policy,fit_end:int):
    """Only original 115 economic predictors; target history stays intact."""
    if fit_end!=TRAIN_END or raw.shape!=(68,59,157):
        raise ValueError("Outlier fit must use only full Stage04 Train")
    values=np.asarray(raw,dtype=np.float64).copy()
    col=values[:,:,SOURCE_FIRST:]
    prior=values[:fit_end,:,SOURCE_FIRST:]
    q=_observed_train_quantile(prior,[.005,.01,.25,.5,.75,.99,.995])
    med=q[3]
    iqr=q[4]-q[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        mad=1.4826*np.nanmedian(np.abs(prior-med),axis=(0,1))
        std=np.nanstd(prior,axis=(0,1))
    if policy.outlier=="none":
        return values
    if policy.outlier in ("winsor005","winsor01"):
        lo,hi=(q[0],q[-1]) if policy.outlier=="winsor005" else (q[1],q[-2])
        col[:]=np.clip(col,lo,hi)
    elif policy.outlier=="mad_clip5":
        s=np.where(mad>1e-8,mad,np.where(std>1e-8,std,1.))
        col[:]=np.clip(col,med-5*s,med+5*s)
    elif policy.outlier in ("asinh","signed_log1p"):
        # Center/scale are determined ONLY by observed Train. Smoothly
        # compress extreme raw-source values; keep original-like units.
        s=np.where(iqr>1e-8,iqr,np.where(std>1e-8,std,1.))
        z=(col-med)/s
        compressed=(np.arcsinh(z) if policy.outlier=="asinh"
                    else np.sign(z)*np.log1p(np.abs(z)))
        col[:]=med+s*compressed
    else:
        raise ValueError(policy.outlier)
    return values


def causal_source_impute(raw,observed,policy:Policy,fit_end:int):
    """Original mask immutable; source channels only. Never inspect X[t+1:]."""
    raw=np.asarray(raw,dtype=np.float64)
    missing=~np.asarray(observed,bool)
    values=raw.copy()
    if policy.missing in ("raw_zero","scaled_zero"):
        return values
    raw_source=values[:,:,SOURCE_FIRST:]
    obs_source=~missing[:,:,SOURCE_FIRST:]
    full_train=raw_source[:fit_end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(full_train,axis=(0,1))
        mean=np.nanmean(full_train,axis=(0,1))
    if policy.missing in ("train_median","train_mean"):
        typical=median if policy.missing=="train_median" else mean
        raw_source[:]=np.where(obs_source,raw_source,typical)
    else:
        # Stateful imputers access actual earlier observations ONLY.
        # When no prior observation exists use a Train-fitted fallback.
        # Do not evaluate a statistic over a whole 2020-22 Dev distribution.
        last=np.full(raw_source.shape[1:],np.nan,dtype=float)
        seasonal=np.full((4,*raw_source.shape[1:]),np.nan,dtype=float)
        for t in range(len(values)):
            good=obs_source[t]
            old=raw_source[t]
            if policy.missing=="causal_ffill":
                proposal=np.where(np.isfinite(last),last,median)
            elif policy.missing=="causal_seasonal":
                earlier=seasonal[t%4]
                proposal=np.where(np.isfinite(earlier),earlier,
                                  np.where(np.isfinite(last),last,median))
            else:
                raise ValueError(policy.missing)
            raw_source[t]=np.where(good,old,proposal)
            last=np.where(good,old,last)
            seasonal[t%4]=np.where(good,old,seasonal[t%4])
    return values


def normalize_inputs(raw,obs,policy:Policy,fit_end):
    """Fit scaler/transform on observed Train only; mask missing separately."""
    train=np.where(obs[:fit_end],raw[:fit_end],np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        med=np.nanmedian(train,axis=(0,1))
        mu=np.nanmean(train,axis=(0,1))
        std=np.nanstd(train,axis=(0,1))
        mad=1.4826*np.nanmedian(np.abs(train-med),axis=(0,1))
        q=_observed_train_quantile(train,[.25,.75])
        maxabs=np.nanmax(np.abs(train),axis=(0,1))
    if policy.normalization=="none":
        center=np.zeros(157)
        width=np.ones(157)
    elif policy.normalization=="standard":
        # Exactly the same observed-only Train standard as Stage04.
        center,width=stage04().fit_normalizer(raw[:fit_end],"standard")
    elif policy.normalization=="robust_iqr":
        center=med
        width=q[1]-q[0]
    elif policy.normalization=="robust_mad":
        center=med
        width=np.where(mad>1e-8,mad,q[1]-q[0])
    elif policy.normalization=="maxabs":
        center=np.zeros(157)
        width=maxabs
    elif policy.normalization=="quantile_normal":
        # QuantileTransformer ignores NaN in fit and uses ONLY Train rows;
        # maintain NaN until after transform to avoid learning imputed modes.
        quantile=QuantileTransformer(
            n_quantiles=min(1000,fit_end*59),output_distribution="normal",
            random_state=20261013,subsample=None)
        quantile.fit(train.reshape(-1,157))
        transformed=quantile.transform(
            np.where(obs,raw,np.nan).reshape(-1,157)).reshape(raw.shape)
        center=np.zeros(157)
        width=np.ones(157)
        # Values imputed by a causal/history method are transformed using
        # the Train-fitted map without refitting it on inserted values.
        if policy.missing not in ("raw_zero","scaled_zero"):
            filled_quantile=quantile.transform(raw.reshape(-1,157)).reshape(raw.shape)
            transformed=np.where(obs,transformed,filled_quantile)
        else:
            replacement=np.zeros_like(transformed)
            if policy.missing=="raw_zero":
                rawzeros=np.where(obs,raw,0.)
                replacement=quantile.transform(
                    rawzeros.reshape(-1,157)).reshape(raw.shape)
            transformed=np.where(obs,transformed,replacement)
        transformed=np.nan_to_num(transformed,nan=0.,posinf=0.,neginf=0.)
        return transformed.astype(np.float32),center,width
    else:
        raise ValueError(policy.normalization)
    center=np.where(np.isfinite(center),center,0.)
    width=np.where(np.isfinite(width)&(width>1e-8),width,1.)
    if policy.missing=="raw_zero":
        filled=np.where(obs,raw,0.)
    elif policy.missing=="scaled_zero":
        filled=np.where(obs,raw,center)
    else:
        # Imputed source values are present, but the target-derived first
        # 42 channels still use the central neutral value.
        filled=np.where(np.isfinite(raw),raw,center)
    normalized=(filled-center)/width
    if (normalized.shape!=(DEV_END,59,157)
            or not np.isfinite(normalized).all()):
        raise RuntimeError("Nonfinite Stage06 normalized input")
    return normalized.astype(np.float32),center,width


def prepare_policy(panel,policy:Policy,prior:np.ndarray,fit_end:int=TRAIN_END):
    """Create Stage04-comparable 12Q windows without any Val distribution fit."""
    policy.validate()
    if (fit_end!=TRAIN_END or panel.x.shape!=(68,59,157)
            or prior.shape!=panel.y.shape):
        raise ValueError("Cannot fit preprocessing on Dev or unequal Stage04 panel")
    raw=np.asarray(panel.x,dtype=np.float64)
    observed=np.isfinite(raw)
    adjusted=outlier_transform(raw,policy,fit_end)
    replaced=causal_source_impute(adjusted,observed,policy,fit_end)
    x,xc,xs=normalize_inputs(replaced,observed,policy,fit_end)
    mask=(~observed).astype(np.float32)
    yc=panel.y[:fit_end].mean((0,1))
    ys=panel.y[:fit_end].std((0,1))
    ys=np.where(ys>1e-8,ys,1.)
    y=((panel.y-yc)/ys).astype(np.float32)
    structured=((prior-yc)/ys).astype(np.float32)
    train=np.arange(LOOKBACK,fit_end,dtype=int)
    dev=np.arange(fit_end,DEV_END,dtype=int)
    m=stage04()
    def pack(ids,prefix):
        return {
            f"{prefix}_x":np.stack([x[t-LOOKBACK:t] for t in ids]),
            f"{prefix}_mask":np.stack([mask[t-LOOKBACK:t] for t in ids]),
            f"{prefix}_y":np.stack([y[t:t+1] for t in ids]),
            f"{prefix}_last":((m.past_scores(panel,ids)-yc)/ys).astype(np.float32),
            f"{prefix}_context":m.known_score_context(panel,ids,yc,ys),
            f"{prefix}_structure":structured[ids].astype(np.float32)}
    data=dict(**pack(train,"train"),**pack(dev,"val"),
              x_center=xc.astype(np.float32),
              x_scale=xs.astype(np.float32),
              y_center=yc.astype(np.float32),
              y_scale=ys.astype(np.float32),
              train_origins=train,val_origins=dev)
    if (data["train_x"].shape!=(44,12,59,157)
            or data["val_x"].shape!=(12,12,59,157)
            or not all(np.isfinite(v).all() for v in data.values())):
        raise RuntimeError("Stage06 data shapes/finite contract invalid")
    audit=dict(
        policy=policy.id,train_fit_end=panel.quarters[fit_end-1],
        distribution_fit_uses_dev=False,
        source_predictors_touched_only=True,
        original_missingness_preserved=np.array_equal(
            data["val_mask"][-1],
            mask[DEV_END-LOOKBACK-1:DEV_END-1]),
        train_original_missing_cells=int((~observed[:fit_end]).sum()),
        original_train_observed_cells=int(observed[:fit_end].sum()),
        raw_test_opened=False,
        input_train_shape=list(data["train_x"].shape),
        input_val_shape=list(data["val_x"].shape))
    if not audit["original_missingness_preserved"]:
        raise RuntimeError("Stage06 corrupted original missing mask")
    return data,audit


def structural_prior(panel):
    m=stage04()
    n,d,info=m.economic_components(panel)
    out=np.zeros_like(panel.y)
    idx=np.arange(LOOKBACK,DEV_END,dtype=int)
    out[idx]=m.component_structural_prior(
        n,d,info,idx,np.asarray(m.STAGE04_FIXED_ALPHA,float))
    if not np.isfinite(out).all():
        raise RuntimeError("Original Stage04 Structural prior failed")
    return out


def original_04_parity(panel,prior):
    """Mandatory bytewise transform check for A=04 reference.

    All 04 normalization, target/context/prior/graph choices are identical
    for the baseline. Later-policy changes affect X values ONLY.
    """
    m=stage04()
    actual,_=prepare_policy(panel,ORIGINAL,prior)
    canonical=m.prepare_three_axis_fold(
        panel,prior,TRAIN_END,TRAIN_END,DEV_END,"standard","mask")
    keys=("train_x","train_mask","train_y","train_last","train_context",
          "train_structure","val_x","val_mask","val_y","val_last",
          "val_context","val_structure",
          "x_center","x_scale","y_center","y_scale","train_origins","val_origins")
    differences=[]
    for key in keys:
        if not np.array_equal(actual[key],canonical[key]):
            differences.append(key)
    if differences:
        raise RuntimeError(
            "Stage06 nominal 04 baseline IS NOT identical: "+str(differences))
    return True


def input_fingerprint(data:dict)->str:
    hasher=hashlib.sha256()
    for key in (
        "train_x","train_mask","train_y","train_last","train_context",
        "train_structure","val_x","val_mask","val_y","val_last",
        "val_context","val_structure","x_center","x_scale",
        "y_center","y_scale","train_origins","val_origins"):
        a=np.ascontiguousarray(data[key])
        hasher.update(key.encode()+b"\0")
        hasher.update(str(a.shape).encode()+str(a.dtype).encode())
        hasher.update(a.tobytes())
    return hasher.hexdigest()


def scores(pred,truth,past):
    pred=np.asarray(pred,np.float64)
    truth=np.asarray(truth,np.float64)
    past=np.asarray(past,np.float64)
    if (pred.shape!=truth.shape or pred.shape!=past.shape
            or pred.shape!=(12,59,3)
            or not all(np.isfinite(z).all() for z in (pred,truth,past))):
        raise RuntimeError("Invalid 12-quarter forecast for Stage06")
    err=pred-truth
    result=dict(
        axis_mae=float(np.abs(err).mean()),
        ici_mae=float(np.abs(err.mean(-1)).mean()),
        persistence_mae=float(np.abs(past-truth).mean()))
    for i,name in enumerate(THREE_AXES):
        result[name+"_mae"]=float(np.abs(err[...,i]).mean())
    for i,year in enumerate((2020,2021,2022)):
        result[f"{year}_mae"]=float(np.abs(err[4*i:4*(i+1)]).mean())
        result[f"{year}_ici_mae"]=float(
            np.abs(err[4*i:4*(i+1)].mean(-1)).mean())
    return result


def evaluate_policy_trials(root:Path,policy:Policy,models,epochs,panel,
                           prior,device,signature,start_time,max_minutes):
    """Resume individual 04-equivalent 5x5 final-Train jobs.

    A changed policy gets no Dev-driven epoch selection. Epoch counts are
    those Stage04 locked on TRAIN-only earlier, before this comparison.
    """
    policy.validate()
    name=policy.id
    results=root/"jobs"/name
    baseline=(policy==ORIGINAL)
    data=None
    if not baseline:
        data,audit=prepare_policy(panel,policy,prior)
        data_hash=input_fingerprint(data)
    else:
        audit=dict(baseline_exact_04_reused=True)
        data_hash="STAGE04_OFFICIAL_LOCKED_ARRAYS"
    preds=[]
    records=[]
    failure=results/"INFEASIBLE.json"
    if failure.is_file():
        state=json_read(failure)
        if (state["signature"]!=signature or state["policy"]!=name
                or state["input_sha256"]!=data_hash):
            raise RuntimeError("Stale infeasibility record in policy cache")
        return dict(
            policy=name,**asdict(policy),status="NUMERICAL_FAILURE",
            error=state["error"],failed_model=state["model"],
            failed_seed=state["seed"],input_sha256=data_hash),None
    for model in models:
        seed_banks=[]
        for seed in SEEDS:
            tag=f"{model}__s{seed}"
            metric=results/(tag+".json")
            file=results/(tag+".npz")
            weight=results/(tag+".pt")
            if baseline:
                ref=RESULT04/"predictions"/(
                    f"structural__{model}__standard__mask__s{seed}.npz")
                with np.load(ref,allow_pickle=False) as z:
                    pred=z["prediction"].copy()
                if pred.shape!=(12,59,3) or not np.isfinite(pred).all():
                    raise RuntimeError(f"Original Stage04 prediction unavailable: {ref}")
                detail=dict(reused_04=True,model=model,seed=seed,
                            epoch=epochs[(model,seed)])
            elif metric.is_file() and file.is_file() and weight.is_file():
                detail=json_read(metric)
                if (detail.get("signature")!=signature
                        or detail.get("policy")!=name
                        or detail.get("model")!=model or detail.get("seed")!=seed
                        or detail.get("epoch")!=epochs[(model,seed)]
                        or detail.get("input_sha256")!=data_hash):
                    raise RuntimeError(f"Different experiment in Stage06 cache: {tag}")
                with np.load(file,allow_pickle=False) as z:
                    pred=z["prediction"].copy()
                if pred.shape!=(12,59,3) or not np.isfinite(pred).all():
                    raise RuntimeError("Invalid saved Stage06 forecast")
                checkpoint=torch.load(weight,map_location="cpu",weights_only=True)
                if (checkpoint["policy"]!=name
                        or checkpoint["signature"]!=signature
                        or checkpoint["model"]!=model or checkpoint["seed"]!=seed
                        or checkpoint["epoch"]!=epochs[(model,seed)]
                        or checkpoint["input_sha256"]!=data_hash):
                    raise RuntimeError("Stage06 checkpoint contradicts job metadata")
            elif metric.exists() or file.exists() or weight.exists():
                raise RuntimeError(
                    f"Interrupted Stage06 partial job {metric}. Resolve or quarantine it first")
            else:
                if time.perf_counter()-start_time>max_minutes*60:
                    return None,dict(next_policy=name,next_model=model,
                                     next_seed=seed,baseline=False)
                # Same Stage04 04.fit_three_axis_gnn with fixed prior,
                # architecture, loss, optimizer, graph and Monte Carlo policy.
                # NEVER call with fixed_epochs=None: no inner/Dev label use
                # for choosing epochs in this stage.
                try:
                    learned=stage04().fit_three_axis_gnn(
                        panel,data,model,seed,device,epochs[(model,seed)],
                        prediction_mode="structural")
                except (RuntimeError,ValueError,FloatingPointError) as exc:
                    error=str(exc)
                    if not any(term in error.lower() for term in (
                            "nonfinite","not finite","nan","infinity","inf detected")):
                        raise
                    # A non-normalized model with unstable magnitudes may
                    # be unusable. Record it, DO NOT hide it and DO NOT
                    # silently drop a failing seed from an ensemble.
                    state=dict(
                        signature=signature,policy=name,
                        input_sha256=data_hash,model=model,seed=seed,
                        error=f"{type(exc).__name__}: {exc}",
                        status="NUMERICAL_FAILURE")
                    results.mkdir(parents=True,exist_ok=True)
                    write_json(failure,state)
                    print("06_CANDIDATE_NUMERICALLY_UNSTABLE",
                          policy.id,model,seed,error,flush=True)
                    return dict(policy=name,**asdict(policy),
                                status="NUMERICAL_FAILURE",error=state["error"],
                                failed_model=model,failed_seed=seed,
                                input_sha256=data_hash),None
                pred=learned["prediction"]
                detail=dict(
                    signature=signature,policy=name,model=model,seed=seed,
                    epoch=epochs[(model,seed)],
                    input_sha256=data_hash,
                    fitted_train_end="2019Q4",
                    used_dev_to_choose_epoch=False,
                    prediction_horizon=1,lookback=LOOKBACK,
                    node_count=59,feature_count=157,
                    fit_seconds=float(learned["seconds"]),
                    inference_regime=learned["inference_regime"],
                    inference_draws=learned["inference_draws"])
                results.mkdir(parents=True,exist_ok=True)
                checkpoint=dict(
                    signature=signature,policy=name,model=model,seed=seed,
                    epoch=epochs[(model,seed)],input_sha256=data_hash,
                    model_state_dict=learned["state_dict"])
                # Store safe checkpoint FIRST, then array, then metric LAST.
                temporary=weight.with_suffix(".pt.writing")
                torch.save(checkpoint,temporary)
                torch.load(temporary,map_location="cpu",weights_only=True)
                os.replace(temporary,weight)
                write_npz(file,prediction=pred)
                write_json(metric,detail)
                print("06_JOB",policy.id,model,seed,
                      "epochs",detail["epoch"],flush=True)
            seed_banks.append(pred)
            records.append(detail)
        preds.append(np.mean(np.stack(seed_banks),axis=0))
    ensemble_prediction=np.mean(np.stack(preds),axis=0)
    original_truth=panel.y[TRAIN_END:DEV_END]
    prior_targets=stage04().past_scores(panel,np.arange(TRAIN_END,DEV_END))
    result=scores(ensemble_prediction,original_truth,prior_targets)
    result.update(policy=name,**asdict(policy),baseline_reused=baseline,
                  num_models=len(models),num_seeds=len(SEEDS),
                  fit_only_train_distribution=True,
                  individual_runs=records,
                  preprocessor_audit=audit,
                  input_sha256=data_hash)
    return dict(**result,prediction=ensemble_prediction),None


def policy_summary(root:Path,policy:Policy,models,epochs,panel,
                   prior,device,signature,start_time,max_minutes):
    location=root/"policies"/(policy.id+".json")
    binary=root/"policies"/(policy.id+".npz")
    if location.is_file():
        result=json_read(location)
        if (result["signature"]!=signature or result["policy"]!=policy.id
                or result["count_gnns"]!=len(models)
                or result["count_seeds"]!=len(SEEDS)):
            raise RuntimeError("Wrong policy summary provenance")
        if result.get("status")=="NUMERICAL_FAILURE":
            if binary.exists():
                raise RuntimeError("Failed policy should not have an ensemble prediction")
            return result,None
        if not binary.is_file():
            raise RuntimeError("Missing prediction for a successful policy")
        with np.load(binary,allow_pickle=False) as z:
            prediction=z["prediction"].copy()
        recomputed=scores(
            prediction,panel.y[TRAIN_END:DEV_END],
            stage04().past_scores(panel,np.arange(TRAIN_END,DEV_END)))
        if any(abs(float(result[k])-v)>1e-10 for k,v in recomputed.items()):
            raise RuntimeError("Saved policy metrics disagree with predictions")
        return result,None
    if binary.exists():
        raise RuntimeError("Incomplete Stage06 policy summary")
    trial,paused=evaluate_policy_trials(
        root,policy,models,epochs,panel,prior,device,signature,
        start_time,max_minutes)
    if paused is not None:
        return None,paused
    forecast=trial.pop("prediction",None)
    result=dict(trial,signature=signature,count_gnns=len(models),
                count_seeds=len(SEEDS))
    if result.get("status")=="NUMERICAL_FAILURE":
        write_json(location,result)
        return result,None
    result["status"]="PASS"
    write_npz(binary,prediction=forecast)
    write_json(location,result)
    return result,None


def greedy_next(current:Policy,stage:int,scores_by_policy:dict):
    """Conservative conditional Stage06 selection, not 216-policy grid."""
    variants=factor_candidates(stage,current)
    if any(p.id not in scores_by_policy for p in variants):
        raise RuntimeError("Stage06 factor cannot choose among incomplete runs")
    base=scores_by_policy[current.id]
    rows=[]
    eligible=[]
    for p in variants:
        entry=scores_by_policy[p.id]
        feasible=entry.get("status")=="PASS"
        year_wins=(sum(
            entry[f"{year}_mae"]<base[f"{year}_mae"]-SELECTION_TOL
            for year in (2020,2021,2022)) if feasible else 0)
        improved=feasible and (entry["axis_mae"] <base["axis_mae"]-SELECTION_TOL)
        ici_safe=feasible and entry["ici_mae"]<=base["ici_mae"]+SELECTION_TOL
        allowed=(p!=current and feasible and improved and ici_safe
                 and year_wins>=MIN_YEARS_BETTER)
        rows.append(dict(
            policy=p.id,stage=stage,component=SLOTS[stage-1],
            **asdict(p),status=entry.get("status"),
            axis_mae=float(entry["axis_mae"]) if feasible else None,
            ici_mae=float(entry["ici_mae"]) if feasible else None,
            delta_mae=(float(entry["axis_mae"]-base["axis_mae"])
                       if feasible else None),
            delta_ici=(float(entry["ici_mae"]-base["ici_mae"])
                       if feasible else None),
            better_years=int(year_wins),
            qualifies_as_change=bool(allowed),
            previously_active=(p==current)))
        if allowed:eligible.append(p)
    winner=min(eligible,key=lambda p:(
        scores_by_policy[p.id]["axis_mae"],
        scores_by_policy[p.id]["ici_mae"],
        p.id)) if eligible else current
    for entry in rows:
        entry["stage_result"]=entry["policy"]==winner.id
    return winner,rows


def ablated_policies(adopted:Policy):
    """Return to original operation one component at a time (up to 3 jobs)."""
    out=[]
    for key in SLOTS:
        before=getattr(ORIGINAL,key)
        after=getattr(adopted,key)
        if after!=before:
            candidate=replace(adopted,**{key:before})
            if candidate.id not in {p.id for p in out}:
                out.append(candidate)
    return tuple(out)


def chart_summaries(root,stage_rows,ablation_rows,all_scores,chosen:Policy,
                    panel,models):
    folder=root/"plots"
    folder.mkdir(parents=True,exist_ok=True)
    outputs=[]
    def save(fig,label):
        fig.tight_layout()
        file=folder/(label+".png")
        fig.savefig(file,dpi=175,bbox_inches="tight")
        plt.close(fig)
        outputs.append("plots/"+file.name)
    grid=pd.DataFrame(stage_rows)
    chosen_steps=[ORIGINAL.id]
    for stage in (1,2,3):
        rows=grid.loc[grid.stage.eq(stage)]
        chosen_steps.append(rows.loc[rows.stage_result,"policy"].iloc[0])
    fig,ax=plt.subplots(figsize=(12,6))
    for stage in (1,2,3):
        rows=grid.loc[grid.stage.eq(stage)]
        y=np.arange(len(rows))
        ax.barh(y+(stage-1)*6.5,rows.axis_mae)
        for j,item in enumerate(rows.itertuples()):
            textlabel=f"{stage}. {getattr(item,SLOTS[stage-1])}"
            if item.stage_result:textlabel+=" (retained)"
            if item.status!="PASS":
                textlabel+=" (numerically unstable)"
            x_text=(float(item.axis_mae)+.005
                    if item.status=="PASS" else
                    float(grid.axis_mae.max())+.005)
            ax.text(x_text,y[j]+(stage-1)*6.5,
                    textlabel,va="center",fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="2020-2022 Dev 3-axis MAE",title="Sequential preprocessing: six candidates per factor")
    save(fig,"06_01_stagewise_mae")

    fig,ax=plt.subplots(figsize=(10.5,5.0))
    vals=[all_scores[p]["axis_mae"] for p in chosen_steps]
    ax.plot(range(4),vals,"o-",lw=2)
    ax.set_xticks(range(4),["Original 04","Missing","Scaling","Outliers"])
    ax.set(ylabel="Dev 3-axis MAE",title="Conservative cumulative preprocessing path")
    ax.grid(alpha=.25)
    save(fig,"06_02_cumulative_mae")

    fig,ax=plt.subplots(figsize=(10.7,5.0))
    years=(2020,2021,2022)
    ax.axhline(0,color="black",lw=1)
    for i,year in enumerate(years):
        diff=(all_scores[chosen.id][f"{year}_mae"]-
              all_scores[ORIGINAL.id][f"{year}_mae"])
        ax.bar(str(year),diff)
    ax.set(ylabel="Selected − Original 04 MAE (negative = better)",
           title="Each Dev year: original 04 vs the policy chosen on Dev")
    ax.grid(axis="y",alpha=.2)
    save(fig,"06_03_year_by_year")

    metrics=["profitability_mae","productivity_mae","growth_mae","ici_mae"]
    a=[all_scores[ORIGINAL.id][k] for k in metrics]
    b=[all_scores[chosen.id][k] for k in metrics]
    xx=np.arange(4)
    fig,ax=plt.subplots(figsize=(10.5,5.0))
    ax.bar(xx-.18,a,width=.36,label="Original 04")
    ax.bar(xx+.18,b,width=.36,label="Selected preprocessing")
    ax.set_xticks(xx,[*THREE_AXES,"ICI"])
    ax.set(ylabel="Dev mean absolute error",title="No target changes; same frozen GNN ensemble")
    ax.legend();ax.grid(axis="y",alpha=.2)
    save(fig,"06_04_three_axes_ici")

    fig,ax=plt.subplots(figsize=(10,4.8))
    if ablation_rows:
        xs=[r["removed_component"] for r in ablation_rows]
        ys=[r["mae_change_on_removal"] for r in ablation_rows]
        ax.bar(xs,ys)
        ax.axhline(0,lw=1,color="black")
    else:
        ax.text(.5,.5,"No changes adopted: no ablation is applicable",
                ha="center",va="center",transform=ax.transAxes)
    ax.set(ylabel="Removed component MAE − final MAE",
           title="Reverse ablation of the final preprocessing combination")
    save(fig,"06_05_final_ablation")
    return outputs


def configuration(m,models,epochs,lock,manifest04,panel):
    state=dict(
        code_sha256=sha(SOURCE),
        stage04_source_sha256=sha(SOURCE04),
        stage04_lock_sha256=sha(RESULT04/"selected_ensemble_pool.json"),
        stage04_manifest_sha256=sha(RESULT04/"run_manifest.json"),
        original_data_sha256=panel.hashes,
        fixed_models=list(models),matched_seeds=list(SEEDS),
        fixed_epochs={f"{model}__s{seed}":int(epochs[(model,seed)])
                      for model in models for seed in SEEDS},
        input_quarters=LOOKBACK,target_horizon_quarters=1,
        output_mode="structural",num_nodes=59,num_features=157,
        structural_coefficients=list(m.STAGE04_FIXED_ALPHA),
        model_selection_epoch_source="original Stage04 Train-internal epochs",
        full_train_end="2019Q4",dev_from="2020Q1",dev_end="2022Q4",
        test_opened=False,no_distribution_fit_from_dev=True,
        dev_reused_for_model_selection=True,
        stage1_missing=list(MISSING),
        stage2_normalization=list(NORMALIZATION),
        stage3_outlier=list(OUTLIER),
        evaluation="actual matched five-GNN five-seed mean ensemble",
        policy_sequence=list(SLOTS),
        minimum_annual_improvement_count=MIN_YEARS_BETTER,
        post_selection_reverse_ablation=True)
    fingerprint=hashlib.sha256(
        json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return state,fingerprint


def _read_policy_prediction(root,policy:Policy):
    path=root/"policies"/(policy.id+".npz")
    with np.load(path,allow_pickle=False) as z:
        forecast=z["prediction"].copy()
    if forecast.shape!=(12,59,3) or not np.isfinite(forecast).all():
        raise RuntimeError("Final Stage06 saved prediction invalid")
    return forecast


def run(args):
    m,panel,models,epochs,lock,manifest04=verify_04_and_load()
    torch.set_num_threads(4)
    spec,signature=configuration(m,models,epochs,lock,manifest04,panel)
    started=time.perf_counter()
    if args.dry_run:
        print("06_SEQUENTIAL_POLICY_DRY_RUN")
        print("MISSING",MISSING)
        print("NORMALIZATION",NORMALIZATION)
        print("OUTLIER",OUTLIER)
        print("SIX PER FACTOR (CURRENT + FIVE ALTERNATIVES)")
        print("3 STAGES: 1 ORIGINAL BASELINE + 5*3 = 16 DISTINCT POLICIES")
        print("FINAL ABLATIONS: AT MOST THREE EXTRA CONDITIONAL POLICIES")
        print("EACH NEW POLICY",len(models),"GNN x",len(SEEDS),"SEEDS")
        print("SOURCE04 EPOCH COUNTS LOCKED; 2006-19 FULL TRAIN; 2020-22 DEV")
        print("SCALER/FILLS/OUTLIER BOUNDS FIT ONLY 2006-19 OBSERVED TRAIN")
        print("STRICTLY LAGGED X[t-12:t], EXACT 04 STRUCTURAL PRIOR")
        print("TEST NEVER OPENED. DEV USED FOR SELECTION (NOT INDEPENDENT TEST)")
        return None
    manifest_file=RESULTS/"run_manifest.json"
    existing=json_read(manifest_file) if manifest_file.is_file() else None
    if args.status:
        counts=list((RESULTS/"policies").glob("*.json"))
        status=(existing or {}).get("status","READY")
        print("06_PREPROCESSING_STATUS",status,
              "POLICIES_SCORED",len(counts),
              "DECISION",(existing or {}).get("selected_policy"),
              "STAGED",(existing or {}).get("current_stage"),
              "TEST_OPENED",False)
        return None
    if not args.run:
        raise RuntimeError("--run is required to fit or resume Stage06")
    if args.max_minutes<=0:
        raise ValueError("Require positive time budget")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if existing and existing.get("signature")!=signature:
        raise RuntimeError(
            "06 code/04 locked upstream/dataset changed: do not mix experiments")
    if existing and existing.get("status")=="PASS":
        print("06_ALREADY_PASS",existing["selected_policy"],flush=True)
        return json_read(RESULTS/"selected_preprocessing.json")
    RESULTS.mkdir(parents=True,exist_ok=True)
    manifest=dict(
        started_utc=(existing or {}).get(
            "started_utc",datetime.now(timezone.utc).isoformat()),
        status="RUNNING",signature=signature,
        contract=spec,models=list(models),seeds=list(SEEDS),
        baseline_policy=ORIGINAL.id,selected_policy=None,
        next_resume="py Sources/06_preprocessing.py --run --max-minutes 25",
        full_train_model_retrained_on_dev=False,
        test_opened=False)
    write_json(manifest_file,manifest)

    # No Train-internal evaluation: all coefficients already locked by the
    # original Stage04 model are fitted entirely BEFORE the first Dev point.
    prior=structural_prior(panel)
    original_04_parity(panel,prior)
    truth=panel.y[TRAIN_END:DEV_END]
    old=original_stage04_prediction(panel,models)
    past=m.past_scores(panel,np.arange(TRAIN_END,DEV_END))
    original_scores=scores(old,truth,past)

    choices=[ORIGINAL]
    evaluated={}
    stage_rows=[]
    ablations=[]
    for stage in (1,2,3):
        incumbent=choices[-1]
        variants=factor_candidates(stage,incumbent)
        print("06_STAGE",stage,"FACTOR",SLOTS[stage-1],
              "CURRENT",incumbent.id,flush=True)
        manifest.update(current_stage=stage,current_factor=SLOTS[stage-1],
                        adopted_so_far=incumbent.id)
        write_json(manifest_file,manifest)
        for policy in variants:
            row,paused=policy_summary(
                RESULTS,policy,models,epochs,panel,prior,device,
                signature,started,args.max_minutes)
            if paused is not None:
                manifest.update(status="PARTIAL_TIME_BUDGET",
                                current_stage=stage,
                                completed_policies=len(list(
                                    (RESULTS/"policies").glob("*.json"))),
                                resume_job=paused,
                                elapsed_seconds=time.perf_counter()-started)
                write_json(manifest_file,manifest)
                print("06_PARTIAL",paused,"MINUTES",args.max_minutes,flush=True)
                return manifest
            evaluated[policy.id]=row
            if row.get("status")=="PASS":
                print("06_POLICY",policy.id,
                      "MAE",round(float(row["axis_mae"]),6),
                      "ICI",round(float(row["ici_mae"]),6),flush=True)
            else:
                print("06_POLICY_INFEASIBLE",policy.id,
                      row.get("error"),flush=True)
        winner,rows=greedy_next(incumbent,stage,evaluated)
        choices.append(winner)
        stage_rows+=rows
        pd.DataFrame(rows).to_csv(
            RESULTS/f"stage{stage:02d}_{SLOTS[stage-1]}_comparison.csv",
            index=False,encoding="utf-8-sig")
        write_json(RESULTS/f"stage{stage:02d}_decision.json",dict(
            step=stage,component=SLOTS[stage-1],
            predecessor=incumbent.id,retained_policy=winner.id,
            method="MAE improves, ICI nonworsens, >=2/3 years improve",
            dev_selection_not_independent_test=True,
            num_candidate_policies=len(variants),
            compared_results=rows,test_opened=False))
        print("06_STAGE_DECISION",stage,winner.id,
              "DELTA_MAE",round(evaluated[winner.id]["axis_mae"]-
                                evaluated[incumbent.id]["axis_mae"],6),
              flush=True)
    adopted=choices[-1]

    for policy in ablated_policies(adopted):
        row,paused=policy_summary(
            RESULTS,policy,models,epochs,panel,prior,device,
            signature,started,args.max_minutes)
        if paused is not None:
            manifest.update(status="PARTIAL_TIME_BUDGET",
                            current_stage=4,resume_job=paused)
            write_json(manifest_file,manifest)
            print("06_PARTIAL_ABLATION",paused,flush=True)
            return manifest
        evaluated[policy.id]=row
        removed=next(slot for slot in SLOTS
                     if getattr(policy,slot)!=getattr(adopted,slot))
        feasible=row.get("status")=="PASS"
        ablations.append(dict(
            removed_component=removed,
            retained_combination=adopted.id,
            without_component=policy.id,
            status=row.get("status"),
            selected_mae=float(evaluated[adopted.id]["axis_mae"]),
            reversed_mae=float(row["axis_mae"]) if feasible else None,
            mae_change_on_removal=(
                float(row["axis_mae"]-evaluated[adopted.id]["axis_mae"])
                if feasible else None),
            selected_ici=float(evaluated[adopted.id]["ici_mae"]),
            reversed_ici=float(row["ici_mae"]) if feasible else None))
    pd.DataFrame(ablations,columns=[
        "removed_component","retained_combination","without_component","status",
        "selected_mae","reversed_mae","mae_change_on_removal",
        "selected_ici","reversed_ici"]).to_csv(
            RESULTS/"final_ablation.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(stage_rows).to_csv(
        RESULTS/"sequential_policy_comparisons.csv",
        index=False,encoding="utf-8-sig")
    chart_list=chart_summaries(
        RESULTS,stage_rows,ablations,evaluated,adopted,panel,models)

    # The submitted operational policy is a FULL Train fitted frozen
    # checkpoint ensemble already saved in jobs/. No new Dev refit.
    chosen_val=_read_policy_prediction(RESULTS,adopted)
    if not np.isfinite(chosen_val).all():
        raise RuntimeError("Selected frozen model prediction invalid")
    write_npz(RESULTS/"frozen_selected_vs_04_dev.npz",
              prediction=chosen_val,
              original_04_prediction=old,
              truth=truth,asof_persistence=past,
              quarters=np.asarray(panel.quarters[TRAIN_END:DEV_END]))
    final_score=scores(chosen_val,truth,past)
    all_scores=pd.DataFrame([
        {key:value for key,value in row.items()
         if key in ("policy","missing","normalization","outlier","axis_mae",
                    "ici_mae","profitability_mae","productivity_mae","growth_mae",
                    "2020_mae","2021_mae","2022_mae")}
        for row in evaluated.values()])
    all_scores.to_csv(RESULTS/"all_policy_scores.csv",
                      index=False,encoding="utf-8-sig")
    selection=dict(
        stage=6,status="LOCKED",policy_id=adopted.id,
        preprocessing=asdict(adopted),
        unchanged_04_reference_policy=asdict(ORIGINAL),
        original_04_dev_metrics=original_scores,
        frozen_selected_dev_metrics=final_score,
        total_policy_lookups=len(evaluated),
        three_factors_six_candidates_per_stage=True,
        each_step_development_selected=True,
        no_heldout_independent_test_claim=True,
        test_opened=False,
        selected_on="Dev 2020Q1-2022Q4; previously used for Stage04 model selection",
        full_train_end="2019Q4",
        no_fitted_distributions_from_dev=True,
        no_train_internal_evaluation=True,
        frozen_weights_for_every_dev_quarter=True,
        gnn_models=list(models),matched_seeds=list(SEEDS),
        input_length_quarters=LOOKBACK,forecast_horizon_quarters=1,
        structural_coefficients=list(m.STAGE04_FIXED_ALPHA),
        reused_04_epoch_counts=True,
        fixed_training_source_sha256=sha(SOURCE04),
        stage04_model_pool_sha256=sha(RESULT04/"selected_ensemble_pool.json"),
        stage06_signature=signature,
        stage_summaries=[
            dict(stage=i,policy=choices[i].id)
            for i in range(1,4)],
        ablation=ablations,
        figures=chart_list,
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(RESULTS/"selected_preprocessing.json",selection)
    manifest.update(
        status="PASS",selected_policy=adopted.id,
        completed_policy_scores=len(evaluated),
        current_stage="COMPLETE",figures=chart_list,
        elapsed_seconds=time.perf_counter()-started,
        completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_file,manifest)
    print("06_SEQUENTIAL_COMPLETE",adopted.id,
          "DEV_MAE",round(final_score["axis_mae"],6),
          "04_MAE",round(original_scores["axis_mae"],6),
          "PLOTS",len(chart_list),flush=True)
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--max-minutes",type=float,default=25.)
    parser.add_argument("--device",choices=("cpu","cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    run(args)


if __name__=="__main__":
    main()
