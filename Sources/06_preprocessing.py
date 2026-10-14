#!/usr/bin/env python3
"""PRISM 06 — MAIN FULL-FACTORIAL PREPROCESSING SEARCH (NO LEGACY MODE).

RESEARCH DESIGN: exhaustively evaluate every 6 x 6 x 6 = 216 registered
combinations of missing-data representation, normalization, and source-
feature distribution/tail transformation. This is ONE canonical full-
factorial experiment. No old Stage06 sequential code, old result readers,
compatibility shims or alternative study versions are supported.

FIXED CONTROLS: Stage04 Structural B-MTGNN and Stage05 Dev-selected raw
feature-window L=1. Every candidate has the SAME 67,190-parameter,
20-slot temporal model, ten matched seeds 2026–2035, fixed 75 epochs,
30 MC-dropout samples, target definitions, original missing mask and
strictly as-of 9D target context. Training uses all 2006–2019 data for
distribution statistics, with the SAME 36 supervised target origins
2011–2019. Dev 2020–2022 has 12 matched evaluation quarters.

POLICY SELECTION: score each seed's Dev three-axis MAE before averaging.
Relative to the Stage05 L1 raw_zero/standard/none baseline, a candidate
is eligible only if mean MAE strictly improves, mean seed-level ICI MAE
does not worsen, and MAE improves in >=2 of 2020/2021/2022. The baseline
is always eligible as fallback. Choose the feasible grid policy with
lowest mean three-axis MAE; ties go to ICI then lexical policy ID.
Thus 'global optimum' means ONLY the constrained, predefined 216-policy
grid on reused Dev—not all possible transforms or future data.

TRANSPARENCY: the 216-way search was chosen after seeing earlier Dev
results and Dev already informed Stage04 and Stage05 selection. The
results are exploratory policy tuning, NOT independent out-of-sample
validation. No statistical significance tests are inferred from ten
seed repeats. Identical transformed tensors are de-duplicated, and
numerically failed policies are explicitly marked infeasible.

FACTORS ARE SEARCHED JOINTLY. Within EACH policy, actual operations
are applied in this order: distribution transform -> source imputation
-> normalization. Outlier winsorization and nonlinear asinh/log1p
compression are distinct levels; nonlinear transforms affect inliers
too. All fitted statistics use observed Train cells ONLY. The first
42 target-history features remain unchanged by source imputation and
distribution transforms. Test data is NEVER opened.

  py Sources/06_preprocessing.py --dry-run
  py Sources/06_preprocessing.py --design-plot
  py Sources/06_preprocessing.py --status
  py -u Sources/06_preprocessing.py --run --max-minutes 25
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
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
SOURCE04=ROOT/"Sources/04_bmtgnn_benchmark.py"
RESULT04=ROOT/"Results/04_bmtgnn_benchmark"
SOURCE05=ROOT/"Sources/05_input_lookback.py"
RESULT05=ROOT/"Results/05_input_lookback"
RESULTS=ROOT/"Results/06_preprocessing"

LOOKBACK=1
TRAIN_END=56
DEV_END=68
SOURCE_FIRST=42
THREE_AXES=("profitability","productivity","growth")
SEEDS=tuple(range(2026,2036))
MODEL="B-MTGNN"
MODE="structural"
TRAIN_ORIGINS=np.arange(20,TRAIN_END,dtype=int)
MIN_YEARS_BETTER=2 # 2020,2021,2022; no 4-year 2016-19 internal evaluation
SELECTION_TOL=1e-9
MISSING=("raw_zero","scaled_zero","train_median","train_mean",
         "causal_ffill","causal_seasonal")
NORMALIZATION=("none","standard","robust_iqr","robust_mad",
               "maxabs","quantile_normal")
# The third factor is broader than outlier *removal*: asinh/log1p also
# transform ordinary inliers, so name it "distribution" throughout.
DISTRIBUTION=("none","winsor005","winsor01","mad_clip5",
              "asinh","signed_log1p")
SLOTS=("missing","normalization","distribution")
GRID_SIZE=len(MISSING)*len(NORMALIZATION)*len(DISTRIBUTION)
assert GRID_SIZE==216


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


@lru_cache(maxsize=1)
def stage05():
    spec=importlib.util.spec_from_file_location("_prism06_stage05",SOURCE05)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Policy:
    missing:str="raw_zero"
    normalization:str="standard"
    distribution:str="none"

    @property
    def id(self)->str:
        return f"m-{self.missing}__n-{self.normalization}__d-{self.distribution}"

    def validate(self):
        if (self.missing not in MISSING
                or self.normalization not in NORMALIZATION
                or self.distribution not in DISTRIBUTION):
            raise ValueError(f"Invalid Stage06 preprocessing {self}")

ORIGINAL=Policy()


def enumerate_policies()->tuple[Policy,...]:
    """Exactly 216 policies; baseline first, then every remaining tuple."""
    grid=tuple(Policy(*axes) for axes in
               product(MISSING,NORMALIZATION,DISTRIBUTION))
    if len(grid)!=216 or len({p.id for p in grid})!=216:
        raise RuntimeError("A full factorial requires 216 unique policies")
    return (ORIGINAL,)+tuple(p for p in grid if p!=ORIGINAL)


def verify_upstream_and_load():
    """Fail closed on selected Stage05 L=1 and the original 200 source jobs."""
    m=stage04()
    s5=stage05()
    canonical_m,panel,lock,manifest04=s5.upstream()
    selection=json_read(RESULT05/"lookback_dev_selection.json")
    run05=json_read(RESULT05/"run_manifest.json")
    if (manifest04.get("status")!="DEV_FROZEN"
            or run05.get("status")!="COMPLETE_MAIN_DEV_ANALYSIS"
            or run05.get("completed_runs")!=200
            or selection.get("selected_lookback_quarters")!=LOOKBACK
            or selection.get("model")!=MODEL
            or selection.get("mode")!=MODE
            or selection.get("seed_count")!=len(SEEDS)
            or selection.get("selection_data")!="DEV_2020Q1_2022Q4"
            or selection.get("selection_rule")!=s5.DEV_SELECTION_RULE
            or run05.get("analysis_source_sha256")!=sha(SOURCE05)
            or run05.get("test_opened") is not False
            or s5.SEEDS!=SEEDS
            ):
        raise RuntimeError("Stage06 requires the current completed L=1 Stage05")
    if (s5.CORE_LOOKBACK!=20 or s5.MODEL!=MODEL
            or s5.MODE!=MODE or tuple(s5.COMMON_TRAIN_ORIGINS)!=tuple(TRAIN_ORIGINS)
            or m.STAGE04_ACTIVE_EPOCHS!=75
            or panel.x.shape!=(68,59,157) or panel.y.shape!=(68,59,3)
            or panel.train_count!=TRAIN_END):
        raise RuntimeError("Stage05 fixed-capacity/current-model contract drift")
    expected={(length,seed) for length in s5.LENGTHS for seed in SEEDS}
    # Verify full completed input bank has exactly the same frozen signature;
    # the baseline reference's 10 NPZ hashes are checked on actual reads.
    metas=list((RESULT05/"trials").glob("*.json"))
    if len(metas)!=len(expected):
        raise RuntimeError("Stage05 trial metadata count differs from 200")
    seen=set()
    for path in metas:
        entry=json_read(path)
        if entry.get("signature")!=run05["signature"]:
            raise RuntimeError("Stage05 trial signature changed")
        seen.add((entry["L"],entry["seed"]))
    if seen!=expected:
        raise RuntimeError("Stage05 trial/seed coverage not complete")
    return m,s5,panel,selection,run05


def original_stage05_seed_predictions(panel):
    """Return 10 individually trained L=1 forecasts without seed averaging."""
    model=stage05()
    stored=json_read(RESULT05/"run_manifest.json")
    bank=[]
    for seed in SEEDS:
        stem=f"L{LOOKBACK:02d}__{MODEL}__s{seed}"
        meta=RESULT05/"trials"/(stem+".json")
        file=RESULT05/"trials"/(stem+".npz")
        row=json_read(meta)
        if (row.get("signature")!=stored["signature"]
                or row.get("L")!=LOOKBACK or row.get("seed")!=seed
                or row.get("model")!=MODEL or
                row.get("fixed_epochs")!=75
                or row.get("prediction_sha256")!=sha(file)):
            raise RuntimeError("Stage05 original selected-L job modified")
        with np.load(file,allow_pickle=False) as z:
            prediction=z["prediction"].copy()
        if (prediction.shape!=(12,59,3) or
                not np.isfinite(prediction).all()):
            raise RuntimeError("Stage05 original L=1 forecast invalid")
        bank.append(prediction)
    return np.stack(bank)


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
            raise RuntimeError("Incomplete Train feature distribution")
    return dict(median=median,mean=mean,std=std,mad=mad,
                maxabs=maxabs,q005=q[0],q01=q[1],
                q25=q[2],q75=q[3],q99=q[4],q995=q[5])


def distribution_transform(raw:np.ndarray,policy:Policy,fit_end:int):
    """Train-fitted source-feature tail/distribution transform (115 cols).

    Nonlinear signed_log1p/asinh compress inliers as well as extremes.
    The first 42 target-history columns are not changed by this function.
    """
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
    if policy.distribution=="none":
        return values
    if policy.distribution in ("winsor005","winsor01"):
        lo,hi=(q[0],q[-1]) if policy.distribution=="winsor005" else (q[1],q[-2])
        col[:]=np.clip(col,lo,hi)
    elif policy.distribution=="mad_clip5":
        s=np.where(mad>1e-8,mad,np.where(std>1e-8,std,1.))
        col[:]=np.clip(col,med-5*s,med+5*s)
    elif policy.distribution in ("asinh","signed_log1p"):
        # Center/scale are determined ONLY by observed Train. Smoothly
        # compress extreme raw-source values; keep original-like units.
        s=np.where(iqr>1e-8,iqr,np.where(std>1e-8,std,1.))
        z=(col-med)/s
        compressed=(np.arcsinh(z) if policy.distribution=="asinh"
                    else np.sign(z)*np.log1p(np.abs(z)))
        col[:]=med+s*compressed
    else:
        raise ValueError(policy.distribution)
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
    """Create Stage05-comparable L1 windows, fit preprocessing on Train only."""
    policy.validate()
    if (fit_end!=TRAIN_END or panel.x.shape!=(68,59,157)
            or prior.shape!=panel.y.shape):
        raise ValueError("Cannot fit preprocessing on Dev or unequal panel")
    raw=np.asarray(panel.x,dtype=np.float64)
    observed=np.isfinite(raw)
    adjusted=distribution_transform(raw,policy,fit_end)
    replaced=causal_source_impute(adjusted,observed,policy,fit_end)
    x,xc,xs=normalize_inputs(replaced,observed,policy,fit_end)
    mask=(~observed).astype(np.float32)
    yc=panel.y[:fit_end].mean((0,1))
    ys=panel.y[:fit_end].std((0,1))
    ys=np.where(ys>1e-8,ys,1.)
    y=((panel.y-yc)/ys).astype(np.float32)
    structured=((prior-yc)/ys).astype(np.float32)
    train=TRAIN_ORIGINS
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
    if (data["train_x"].shape!=(36,1,59,157)
            or data["val_x"].shape!=(12,1,59,157)
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
    prior,_=stage04().make_fold(panel)
    if prior.shape!=panel.y.shape or not np.isfinite(prior).all():
        raise RuntimeError("Stage04 Structural Train prior changed")
    return prior


def original_stage05_parity(panel,prior):
    """Bytewise parity with L1 Train/Dev tensors used to fit Stage05.

    If this fails the original policy cannot be reused; do not compare
    unpaired training protocols or silently re-fit the L=1 baseline.
    """
    actual,_=prepare_policy(panel,ORIGINAL,prior)
    canonical=stage05().Inputs(panel).fold(LOOKBACK)
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
            "Stage06 nominal Stage05 L1 baseline IS NOT identical: "
            +str(differences))
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


def fit_policy_model(data,seed,device):
    """Same 75-epoch L1/B-MTGNN recipe and identical seed init as Stage05."""
    m=stage04()
    module=stage05()
    m.seed_all(seed)
    net=module.instantiate(m,MODEL,seed,LOOKBACK,data,MODE).to(device)
    z={k:torch.as_tensor(data[k],device=device)
       for k in ("train_x","train_mask","train_y","train_last",
                 "train_context","train_structure","val_x","val_mask",
                 "val_last","val_context","val_structure")}
    graph=torch.eye(m.NODE_COUNT,device=device)
    scale=torch.as_tensor(data["y_scale"],device=device)
    optim=torch.optim.AdamW(
        net.parameters(),lr=m.FIXED_LEARNING_RATE,weight_decay=1e-4)
    begin=time.perf_counter()
    for _ in range(m.STAGE04_ACTIVE_EPOCHS):
        net.train()
        optim.zero_grad(set_to_none=True)
        predicted=net(z["train_x"],z["train_mask"],graph,
                      z["train_last"],z["train_structure"],
                      z["train_context"])
        loss=m.three_axis_weighted_huber(
            predicted,z["train_y"],z["train_last"],scale)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Stage06 Train loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,
                                        error_if_nonfinite=True)
        optim.step()
    with m.isolated_snapshot_inference(net):
        forecast=m.mtgnn_bayesian_inference(
            net,z,graph,MODEL,seed,
            data["y_center"],data["y_scale"])["prediction"]
    if forecast.shape!=(12,59,3) or not np.isfinite(forecast).all():
        raise RuntimeError("nonfinite Stage06 Dev forecast")
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    seconds=float(time.perf_counter()-begin)
    del net,optim,z
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return forecast,state,seconds


def seed_policy_metrics(predictions,panel):
    predictions=np.asarray(predictions,dtype=np.float64)
    if predictions.shape!=(len(SEEDS),12,59,3):
        raise RuntimeError("Missing one of the ten matched B-MTGNN seeds")
    truth=panel.y[TRAIN_END:DEV_END]
    prior=stage04().past_scores(panel,np.arange(TRAIN_END,DEV_END))
    individual=[scores(pred,truth,prior) for pred in predictions]
    average={key:float(np.mean([entry[key] for entry in individual]))
             for key in individual[0]}
    average["axis_mae_seed_sd"]=float(
        np.std([entry["axis_mae"] for entry in individual],ddof=1))
    average["ici_mae_seed_sd"]=float(
        np.std([entry["ici_mae"] for entry in individual],ddof=1))
    return average,individual


def evaluate_policy_trials(root,policy,panel,prior,device,signature,
                           start_time,max_minutes,input_cache):
    """Evaluate matched individual seeds, reusing byte-identical inputs."""
    policy.validate()
    data,audit=prepare_policy(panel,policy,prior)
    input_sha=input_fingerprint(data)
    equivalent=input_cache.get(input_sha)
    if equivalent is not None and equivalent!=policy.id:
        file=root/"policies"/(equivalent+".npz")
        with np.load(file,allow_pickle=False) as bank:
            forecasts=bank["seed_predictions"].copy()
        print("06_IDENTICAL_INPUT_REUSED",policy.id,"FROM",equivalent,flush=True)
        metric,individual=seed_policy_metrics(forecasts,panel)
        return dict(policy=policy.id,**asdict(policy),status="PASS",
                    **metric,seed_results=individual,
                    input_sha256=input_sha,
                    identical_input_reused_from=equivalent,
                    baseline_from_stage05=False,
                    preprocessor_audit=audit),forecasts,None

    if policy==ORIGINAL:
        original_stage05_parity(panel,prior)
        forecasts=original_stage05_seed_predictions(panel)
        print("06_BASELINE_REUSED_EXACT_STAGE05_L1",flush=True)
        base=True
    else:
        forecasts=[]
        base=False
        jobs=root/"jobs"/policy.id
        failure=jobs/"INFEASIBLE.json"
        if failure.exists():
            existing=json_read(failure)
            if (existing.get("signature")!=signature or
                    existing.get("input_sha256")!=input_sha):
                raise RuntimeError("Stage06 failed-policy cache changed")
            return dict(policy=policy.id,**asdict(policy),
                        status="NUMERICAL_FAILURE",
                        input_sha256=input_sha,
                        failed_seed=existing["seed"],
                        error=existing["error"]),None,None
        for seed in SEEDS:
            stem=f"{MODEL}__s{seed}"
            meta=jobs/(stem+".json")
            predfile=jobs/(stem+".npz")
            weightfile=jobs/(stem+".pt")
            if all(file.is_file() for file in (meta,predfile,weightfile)):
                record=json_read(meta)
                if (record.get("signature")!=signature or
                        record.get("policy")!=policy.id or
                        record.get("seed")!=seed or
                        record.get("model")!=MODEL or
                        record.get("epoch")!=75 or
                        record.get("input_sha256")!=input_sha or
                        record.get("prediction_sha256")!=sha(predfile) or
                        record.get("checkpoint_sha256")!=sha(weightfile)):
                    raise RuntimeError("Stage06 cached job provenance mismatch")
                with np.load(predfile,allow_pickle=False) as saved:
                    forecast=saved["prediction"].copy()
                if forecast.shape!=(12,59,3) or not np.isfinite(forecast).all():
                    raise RuntimeError("Stage06 cached forecast malformed")
                checkpoint=torch.load(
                    weightfile,map_location="cpu",weights_only=True)
                if (checkpoint.get("signature")!=signature or
                        checkpoint.get("policy")!=policy.id or
                        checkpoint.get("seed")!=seed or
                        checkpoint.get("model")!=MODEL or
                        checkpoint.get("epoch")!=75 or
                        checkpoint.get("input_sha256")!=input_sha):
                    raise RuntimeError("Stage06 checkpoint metadata drift")
            elif any(file.exists() for file in (meta,predfile,weightfile)):
                raise RuntimeError("Stage06 interrupted/incomplete job: "+stem)
            else:
                if time.perf_counter()-start_time>=max_minutes*60:
                    return None,None,dict(next_policy=policy.id,next_seed=seed)
                try:
                    forecast,state,seconds=fit_policy_model(data,seed,device)
                except (RuntimeError,ValueError,FloatingPointError) as error:
                    if not any(word in str(error).lower()
                               for word in ("nonfinite","nan","infinity",
                                            "not finite","inf detected")):
                        raise
                    reason=f"{type(error).__name__}: {error}"
                    write_json(failure,dict(
                        signature=signature,policy=policy.id,
                        input_sha256=input_sha,model=MODEL,seed=seed,
                        error=reason,status="NUMERICAL_FAILURE"))
                    print("06_NUMERICAL_FAILURE",policy.id,seed,reason,
                          flush=True)
                    return dict(policy=policy.id,**asdict(policy),
                                status="NUMERICAL_FAILURE",
                                input_sha256=input_sha,
                                failed_seed=seed,error=reason),None,None
                checkpoint=dict(
                    signature=signature,policy=policy.id,seed=seed,
                    model=MODEL,input_sha256=input_sha,epoch=75,
                    model_state_dict=state)
                jobs.mkdir(parents=True,exist_ok=True)
                writing=weightfile.with_suffix(".pt.writing")
                torch.save(checkpoint,writing)
                os.replace(writing,weightfile)
                write_npz(predfile,prediction=forecast)
                write_json(meta,dict(
                    signature=signature,policy=policy.id,seed=seed,
                    model=MODEL,epoch=75,input_sha256=input_sha,
                    prediction_sha256=sha(predfile),
                    checkpoint_sha256=sha(weightfile),
                    fit_seconds=seconds,
                    test_opened=False))
                print("06_JOB",policy.id,"SEED",seed,
                      "EPOCHS 75","SECONDS",round(seconds,2),flush=True)
            forecasts.append(forecast)
        forecasts=np.stack(forecasts)
    metrics,individual=seed_policy_metrics(forecasts,panel)
    return dict(policy=policy.id,**asdict(policy),status="PASS",
                **metrics,seed_results=individual,
                input_sha256=input_sha,
                identical_input_reused_from=None,
                baseline_from_stage05=base,
                preprocessor_audit=audit),forecasts,None


def policy_summary(root,policy,panel,prior,device,signature,
                   start_time,max_minutes,input_cache):
    """Hash-validate frozen seed arrays and resume a policy idempotently."""
    location=root/"policies"/(policy.id+".json")
    binary=root/"policies"/(policy.id+".npz")
    data,audit=prepare_policy(panel,policy,prior)
    input_sha=input_fingerprint(data)
    if location.is_file():
        row=json_read(location)
        if (row.get("signature")!=signature or
                row.get("policy")!=policy.id or
                row.get("input_sha256")!=input_sha or
                row.get("count_seeds")!=len(SEEDS)):
            raise RuntimeError("Stage06 policy signature/input changed")
        if row.get("status")=="NUMERICAL_FAILURE":
            if binary.exists():
                raise RuntimeError("Failed policy has prediction file")
            return row,None
        if not binary.exists() or row.get("prediction_sha256")!=sha(binary):
            raise RuntimeError("Policy prediction cache mismatch")
        with np.load(binary,allow_pickle=False) as saved:
            preds=saved["seed_predictions"].copy()
        computed,_=seed_policy_metrics(preds,panel)
        if any(abs(row[key]-val)>1e-9 for key,val in computed.items()):
            raise RuntimeError("Cached per-seed policy mean differs from metrics")
        input_cache.setdefault(input_sha,policy.id)
        return row,None
    if binary.exists():
        raise RuntimeError("Stage06 incomplete policy report")
    row,forecasts,paused=evaluate_policy_trials(
        root,policy,panel,prior,device,signature,
        start_time,max_minutes,input_cache)
    if paused is not None:
        return None,paused
    row.update(signature=signature,count_seeds=len(SEEDS))
    if row["status"]=="PASS":
        write_npz(binary,seed_predictions=forecasts,
                  prediction=forecasts.mean(0))
        row["prediction_sha256"]=sha(binary)
        input_cache.setdefault(input_sha,policy.id)
    write_json(location,row)
    return row,None


def eligible_policy_table(policies, rows):
    """Score every one of the 216 grid cells against ONE fixed L1 baseline.

    The acceptance criteria are policy-selection heuristics, NOT tests of
    statistical significance or future-year generalization.
    """
    expected={p.id for p in enumerate_policies()}
    if len(policies)!=GRID_SIZE or set(rows)!=expected:
        raise RuntimeError("Cannot select from an incomplete 216-policy grid")
    original=rows[ORIGINAL.id]
    if original.get("status")!="PASS":
        raise RuntimeError("Stage05 L1 reference must have a PASS score")
    records=[]
    for policy in policies:
        result=rows[policy.id]
        passed=result.get("status")=="PASS"
        improved_years=(sum(
            result[f"{year}_mae"]<
            original[f"{year}_mae"]-SELECTION_TOL
            for year in (2020,2021,2022)) if passed else 0)
        mae_better=bool(passed and
                        result["axis_mae"]<
                        original["axis_mae"]-SELECTION_TOL)
        ici_safe=bool(passed and
                      result["ici_mae"]<=
                      original["ici_mae"]+SELECTION_TOL)
        eligible=bool(policy==ORIGINAL or (
            mae_better and ici_safe and improved_years>=MIN_YEARS_BETTER))
        records.append(dict(
            policy=policy.id,**asdict(policy),
            status=result.get("status"),
            axis_mae=float(result["axis_mae"]) if passed else np.nan,
            ici_mae=float(result["ici_mae"]) if passed else np.nan,
            axis_mae_seed_sd=(
                float(result["axis_mae_seed_sd"]) if passed else np.nan),
            ici_mae_seed_sd=(
                float(result["ici_mae_seed_sd"]) if passed else np.nan),
            profitability_mae=(
                float(result["profitability_mae"]) if passed else np.nan),
            productivity_mae=(
                float(result["productivity_mae"]) if passed else np.nan),
            growth_mae=(
                float(result["growth_mae"]) if passed else np.nan),
            **{f"{year}_mae":(
                float(result[f"{year}_mae"]) if passed else np.nan)
               for year in (2020,2021,2022)},
            **{f"{year}_ici_mae":(
                float(result[f"{year}_ici_mae"]) if passed else np.nan)
               for year in (2020,2021,2022)},
            delta_mae_from_original=(
                float(result["axis_mae"]-original["axis_mae"])
                if passed else np.nan),
            delta_ici_from_original=(
                float(result["ici_mae"]-original["ici_mae"])
                if passed else np.nan),
            better_dev_years=int(improved_years),
            beats_primary_baseline=mae_better,
            ici_safeguard_pass=ici_safe,
            eligible=eligible,
            original_reference=(policy==ORIGINAL),
            input_sha256=result.get("input_sha256"),
            identical_input_reused_from=result.get(
                "identical_input_reused_from")))
    table=pd.DataFrame(records)
    if (len(table)!=GRID_SIZE or table.policy.nunique()!=GRID_SIZE
            or not table.loc[table.original_reference,"eligible"].all()
            or table.eligible.sum()<1):
        raise RuntimeError("Stage06 factorial scoring invariant failed")
    return table


def select_grid_policy(table):
    """Constrained GRID optimum, not a universal preprocessing optimum."""
    candidates=table[table.eligible].sort_values(
        ["axis_mae","ici_mae","policy"],kind="stable")
    if candidates.empty or candidates.axis_mae.isna().any():
        raise RuntimeError("No feasible policy to select")
    return Policy(**{
        key:str(candidates.iloc[0][key]) for key in SLOTS})


def descriptive_pareto(table):
    """Nondominated MAE/ICI points among all numerically feasible policies."""
    frame=table[table.status.eq("PASS")].sort_values(
        ["axis_mae","ici_mae","policy"]).copy()
    incumbent=float("inf")
    kept=[]
    for row in frame.itertuples():
        if row.ici_mae<incumbent-1e-12:
            kept.append(row.policy)
            incumbent=float(row.ici_mae)
    return frame[frame.policy.isin(kept)].copy()


def pairwise_interaction_tables(table):
    """Descriptive balanced-grid pair residuals, not causal ANOVA p-values.

    If any policy fails numerically, its cells are NaN; report the
    incomplete-grid mean as descriptive only, never a balanced ANOVA.
    """
    complete=table[table.status.eq("PASS")].copy()
    full_mean=float(complete.axis_mae.mean())
    records=[]
    for first,second in (
        ("missing","normalization"),
        ("missing","distribution"),
        ("normalization","distribution")):
        mu_first=complete.groupby(first).axis_mae.mean()
        mu_second=complete.groupby(second).axis_mae.mean()
        for a in (MISSING if first=="missing" else
                  NORMALIZATION if first=="normalization" else DISTRIBUTION):
            for b in (MISSING if second=="missing" else
                      NORMALIZATION if second=="normalization" else DISTRIBUTION):
                block=complete.loc[complete[first].eq(a)&
                                   complete[second].eq(b)]
                available=len(block)
                residual=(float(block.axis_mae.mean())-
                          float(mu_first[a])-float(mu_second[b])+full_mean
                          if available==6 else np.nan)
                records.append(dict(
                    factor_a=first,level_a=a,factor_b=second,level_b=b,
                    observed_third_factor_levels=available,
                    grid_complete_at_pair=(available==6),
                    two_factor_residual_mae=residual))
    return pd.DataFrame(records)


def save_plot(fig,folder,label,*,tight=True):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(label+".png")
    if tight:
        fig.tight_layout()
    fig.savefig(path,dpi=175,bbox_inches="tight")
    plt.close(fig)
    return "plots/"+path.name


def plot_design(root):
    """Actual registered 216-cell grid, no performance claim."""
    target=root/"plots"
    grid=enumerate_policies()
    if len(grid)!=216:
        raise RuntimeError("Registered factorial design incomplete")
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    for axis,dist in zip(axes.flat,DISTRIBUTION):
        occupancy=np.array([[sum(
            1 for p in grid if
            p.missing==miss and p.normalization==norm and
            p.distribution==dist) for norm in NORMALIZATION]
            for miss in MISSING])
        axis.imshow(occupancy,vmin=0,vmax=2,cmap="Blues",aspect="auto")
        for i in range(6):
            for j in range(6):
                axis.text(j,i,str(occupancy[i,j]),
                          ha="center",va="center",fontsize=8)
        axis.set_title("Distribution: "+dist)
        axis.set_xticks(range(6),NORMALIZATION,rotation=50,ha="right")
        axis.set_yticks(range(6),MISSING)
    fig.suptitle("Registered PRISM06 full factorial design: 216 cells (1 per cell)")
    return save_plot(fig,target,"06_00_registered_216_grid")


def chart_full_factorial(root,table,rows,chosen,panel):
    """Full factorial results plots (all descriptive; no p-values)."""
    folder=root/"plots"
    output=[]
    output.append(plot_design(root))
    base=table.set_index("policy").loc[ORIGINAL.id]
    winner=table.set_index("policy").loc[chosen.id]
    ranked=table.sort_values(["axis_mae","ici_mae"],na_position="last")
    valid=ranked.status.eq("PASS").to_numpy()
    fig,ax=plt.subplots(figsize=(12,5.3))
    xx=np.arange(len(ranked))
    ax.scatter(xx[valid],ranked.loc[valid,"axis_mae"],
               s=20,c=np.where(ranked.loc[valid,"eligible"],
                               "tab:green","0.55"),
               alpha=.65,label="Eligible green / other gray")
    i_winner=ranked.index[ranked.policy.eq(chosen.id)].tolist()[0]
    idx_winner=np.where(ranked.policy.to_numpy()==chosen.id)[0][0]
    idx_baseline=np.where(ranked.policy.to_numpy()==ORIGINAL.id)[0][0]
    ax.scatter([idx_winner],[winner.axis_mae],marker="*",s=160,
               color="tab:red",label="Grid-selected policy",zorder=5)
    ax.scatter([idx_baseline],[base.axis_mae],marker="D",s=65,
               color="tab:orange",label="Stage05 baseline",zorder=5)
    ax.set(xlabel="216 policies sorted by mean Dev three-axis MAE",
           ylabel="Mean individual-seed three-axis Dev MAE",
           title="Full-grid 216 policy distribution and constrained selection")
    ax.grid(alpha=.2);ax.legend()
    output.append(save_plot(fig,folder,"06_01_all_216_ranked_mae"))

    vmin=float(table.axis_mae.min())
    vmax=float(table.axis_mae.max())
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    for ax,d in zip(axes.flat,DISTRIBUTION):
        block=table[table.distribution.eq(d)]
        matrix=block.pivot(index="missing",columns="normalization",
                           values="axis_mae").reindex(
            index=MISSING,columns=NORMALIZATION).to_numpy()
        image=ax.imshow(matrix,cmap="viridis",vmin=vmin,vmax=vmax,
                        aspect="auto")
        if chosen.distribution==d:
            ax.scatter([NORMALIZATION.index(chosen.normalization)],
                       [MISSING.index(chosen.missing)],
                       marker="*",color="red",s=175)
        ax.set_title("Distribution: "+d)
        ax.set_xticks(range(6),NORMALIZATION,rotation=55,ha="right")
        ax.set_yticks(range(6),MISSING)
    fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.025,
                 label="Mean seed-level Dev 3-axis MAE")
    fig.suptitle("Full factorial: six 6 × 6 raw Dev MAE slices")
    output.append(save_plot(fig,folder,"06_02_six_mae_heatmaps",
                            tight=False))

    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,(factor,labels) in zip(axes,(
            ("missing",MISSING),("normalization",NORMALIZATION),
            ("distribution",DISTRIBUTION))):
        means=table.groupby(factor).axis_mae.mean().reindex(labels)
        ax.bar(np.arange(6),means)
        ax.axhline(float(base.axis_mae),ls="--",color="tab:orange",
                   label="Original L1")
        ax.set_xticks(range(6),labels,rotation=55,ha="right")
        ax.set(title=factor,ylabel="Marginal mean MAE over other factors")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Full-grid marginal summaries (not causal main effects)")
    output.append(save_plot(fig,folder,"06_03_marginal_factor_means"))

    interactions=pairwise_interaction_tables(table)
    interactions.to_csv(root/"factor_interactions.csv",
                        index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    magnitude=np.nanmax(np.abs(
        interactions.two_factor_residual_mae.to_numpy()))
    magnitude=max(float(magnitude) if np.isfinite(magnitude) else 0.,1e-10)
    for ax,(aa,bb) in zip(axes,(
            ("missing","normalization"),("missing","distribution"),
            ("normalization","distribution"))):
        entry=interactions[(interactions.factor_a==aa)&
                            (interactions.factor_b==bb)]
        al=MISSING if aa=="missing" else NORMALIZATION
        bl=NORMALIZATION if bb=="normalization" else DISTRIBUTION
        matrix=entry.pivot(index="level_a",columns="level_b",
                           values="two_factor_residual_mae").reindex(
            index=al,columns=bl).to_numpy()
        img=ax.imshow(matrix,cmap="coolwarm",
                      vmin=-magnitude,vmax=magnitude,aspect="auto")
        ax.set_xticks(range(6),bl,rotation=55,ha="right")
        ax.set_yticks(range(6),al)
        ax.set_title(aa+" × "+bb)
    fig.colorbar(img,ax=axes.ravel().tolist(),fraction=.025,
                 label="Pair residual mean MAE (descriptive)")
    fig.suptitle("Full-grid two-factor deviations from additive marginal means")
    output.append(save_plot(fig,folder,"06_04_pairwise_interactions",
                            tight=False))

    fig,ax=plt.subplots(figsize=(10.8,6))
    successful=table[table.status.eq("PASS")]
    for label,subset in successful.groupby("distribution"):
        ax.scatter(subset.axis_mae,subset.ici_mae,s=20,
                   alpha=.55,label=label)
    front=descriptive_pareto(table).sort_values("axis_mae")
    front.to_csv(root/"mae_ici_pareto.csv",
                 index=False,encoding="utf-8-sig")
    ax.plot(front.axis_mae,front.ici_mae,
            "-",color="black",lw=1.5,label="Observed Pareto boundary")
    ax.scatter([winner.axis_mae],[winner.ici_mae],
               marker="*",s=200,color="red",label="Selected",zorder=6)
    ax.scatter([base.axis_mae],[base.ici_mae],
               marker="D",s=75,color="tab:orange",
               label="Stage05 baseline",zorder=6)
    ax.axhline(float(base.ici_mae),color="tab:orange",ls=":")
    ax.set(xlabel="Dev 3-axis MAE",ylabel="Dev ICI MAE",
           title="MAE–ICI tradeoffs in the complete candidate grid")
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    output.append(save_plot(fig,folder,"06_05_mae_ici_pareto"))

    old=np.array([r["axis_mae"] for r in rows[ORIGINAL.id]["seed_results"]])
    current=np.array([r["axis_mae"] for r in rows[chosen.id]["seed_results"]])
    old_ici=np.array([r["ici_mae"] for r in rows[ORIGINAL.id]["seed_results"]])
    current_ici=np.array([r["ici_mae"] for r in rows[chosen.id]["seed_results"]])
    fig,axes=plt.subplots(2,1,figsize=(11.8,7),sharex=True)
    for ax,values,label in zip(axes,(
            current-old,current_ici-old_ici),
            ("Three-axis MAE","ICI MAE")):
        ax.bar(np.arange(10),values,color=[
            "tab:blue" if x<=0 else "tab:orange" for x in values])
        ax.axhline(0,color="black",lw=1)
        ax.set(ylabel="Selected − baseline "+label)
        ax.grid(axis="y",alpha=.2)
    axes[-1].set_xticks(np.arange(10),SEEDS)
    axes[-1].set_xlabel("Matched B-MTGNN training seed")
    fig.suptitle("Selected-policy matched seed changes (descriptive)")
    output.append(save_plot(fig,folder,"06_06_matched_seed_deltas"))

    fig,axes=plt.subplots(1,2,figsize=(14.5,5))
    for ax,keys,labels,title in (
        (axes[0],[f"{year}_mae" for year in (2020,2021,2022)],
         ["2020","2021","2022"],"Per-year MAE"),
        (axes[1],[f"{a}_mae" for a in THREE_AXES]+["ici_mae"],
         [*THREE_AXES,"ICI"],"Three targets and ICI")):
        change=[float(winner[k]-base[k]) for k in keys]
        ax.bar(np.arange(len(change)),change,color=[
            "tab:blue" if x<=0 else "tab:orange" for x in change])
        ax.axhline(0,color="black",lw=1)
        ax.set_xticks(np.arange(len(change)),labels,rotation=25)
        ax.set(ylabel="Selected − baseline Dev MAE",title=title)
        ax.grid(axis="y",alpha=.2)
    output.append(save_plot(fig,folder,"06_07_year_and_target_changes"))

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    eligible=table.pivot_table(
        index="missing",columns="normalization",values="eligible",
        aggfunc="sum").reindex(index=MISSING,columns=NORMALIZATION)
    im=axes[0].imshow(eligible.to_numpy(),vmin=0,vmax=6,
                      cmap="Blues",aspect="auto")
    axes[0].set_title("Eligible policies across distribution levels (out of 6)")
    axes[0].set_xticks(range(6),NORMALIZATION,rotation=45,ha="right")
    axes[0].set_yticks(range(6),MISSING)
    fig.colorbar(im,ax=axes[0],fraction=.045)
    groups=table.groupby("distribution").eligible.sum().reindex(DISTRIBUTION)
    axes[1].bar(range(6),groups)
    axes[1].set_xticks(range(6),DISTRIBUTION,rotation=45,ha="right")
    axes[1].set(title="Feasible policy count by distribution transform",
                ylabel="Count of eligible grid cells (of 36)")
    axes[1].grid(axis="y",alpha=.2)
    output.append(save_plot(fig,folder,"06_08_constraint_eligibility",
                            tight=False))

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,(factor,options) in zip(
            axes,(("missing",MISSING),("normalization",NORMALIZATION),
                  ("distribution",DISTRIBUTION))):
        subset=table.copy()
        for other in SLOTS:
            if other!=factor:
                subset=subset[subset[other].eq(getattr(chosen,other))]
        subset=subset.set_index(factor).reindex(options)
        xx=np.arange(6)
        ax.plot(xx,subset.axis_mae,"o-",label="Conditional Dev MAE")
        ax.axhline(float(winner.axis_mae),color="tab:green",ls="--",
                   label="Selected policy")
        ax.set_xticks(xx,options,rotation=55,ha="right")
        ax.set(title="Vary "+factor+" around chosen policy",
               ylabel="Dev 3-axis MAE")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Conditional factor slices expose interactions; not a greedy search")
    output.append(save_plot(fig,folder,"06_09_conditional_factor_slices"))

    seed_rows=[]
    for pol in table[table.status.eq("PASS")].policy:
        for seed,metric in zip(SEEDS,rows[pol]["seed_results"]):
            seed_rows.append(dict(policy=pol,seed=seed,
                                  axis_mae=metric["axis_mae"],
                                  ici_mae=metric["ici_mae"]))
    seed_frame=pd.DataFrame(seed_rows)
    seed_frame.to_csv(root/"per_seed_policy_scores.csv",
                      index=False,encoding="utf-8-sig")
    seed_frame["within_seed_rank"]=seed_frame.groupby("seed").axis_mae.rank(
        method="min",ascending=True)
    fig,ax=plt.subplots(figsize=(10,5))
    for pol,color,label in (
            (ORIGINAL.id,"tab:orange","Original L1"),
            (chosen.id,"tab:blue","Grid-selected")):
        target=seed_frame[seed_frame.policy.eq(pol)].sort_values("seed")
        ax.plot(target.seed,target.within_seed_rank,"o-",
                color=color,label=label)
    ax.set(xlabel="Training seed",ylabel="Rank among successful grid policies",
           title="Per-seed stability across 216 policy labels (lower is better)")
    ax.grid(alpha=.2);ax.legend()
    output.append(save_plot(fig,folder,"06_10_seed_rank_stability"))
    return output


def configuration(m,m5,selection05,manifest05,panel):
    """One canonical FULL-FACTORIAL signature: no older 06 result support."""
    baselines={}
    for seed in SEEDS:
        stem=f"L{LOOKBACK:02d}__{MODEL}__s{seed}.npz"
        baselines[str(seed)]=sha(RESULT05/"trials"/stem)
    contract=dict(
        experiment="PRISM06_FULL_FACTORIAL_216_SINGLE_CANONICAL",
        source_sha256=sha(SOURCE),
        source04_sha256=sha(SOURCE04),
        source05_sha256=sha(SOURCE05),
        upstream_stage04_manifest_sha256=sha(RESULT04/"run_manifest.json"),
        upstream_stage05_manifest_sha256=sha(RESULT05/"run_manifest.json"),
        upstream_stage05_selection_sha256=sha(
            RESULT05/"lookback_dev_selection.json"),
        stage05_L1_baseline_npz_sha256=baselines,
        original_train_dev_data_sha256=panel.hashes,
        model=MODEL,output_mode=MODE,lookback=LOOKBACK,
        original_reference_lookback=selection05["reference_lookback_quarters"],
        fixed_temporal_core=m5.CORE_LOOKBACK,
        seed_grid=list(SEEDS),fixed_epochs=m.STAGE04_ACTIVE_EPOCHS,
        mc_dropout_draws=m.STAGE04_BMTGNN_MC_SAMPLES,
        train_origin_indices=TRAIN_ORIGINS.tolist(),
        dev_origin_indices=list(range(TRAIN_END,DEV_END)),
        all_train_through="2019Q4",dev_from="2020Q1",dev_end="2022Q4",
        no_train_cv=True,train_fits_only=True,test_opened=False,
        factor_names=list(SLOTS),missing_levels=list(MISSING),
        normalization_levels=list(NORMALIZATION),
        distribution_levels=list(DISTRIBUTION),
        cartesian_policy_count=GRID_SIZE,
        transform_application_order=[
            "distribution","missing","normalization"],
        baseline_id=ORIGINAL.id,
        eligibility_rule=(
            "BASELINE_OR_AXIS_MAE_STRICTLY_IMPROVES_AND_ICI_NOT_WORSE_"
            "AND_AT_LEAST_TWO_OF_THREE_DEV_YEARS_IMPROVE"),
        selection_rule="MIN_10_SEED_AVERAGE_AXIS_MAE_THEN_ICI_THEN_ID",
        score_semantics="SEED_WISE_MAE_BEFORE_SEED_AVERAGE",
        dedupe="IDENTICAL_COMPLETE_MODEL_INPUT_SHA256",
        factorial_interactions="DESCRIPTIVE_NO_PVALUES",
        dev_reused_in_stage04_and_stage05=True,
        full_grid_chosen_after_prior_dev_inspection=True,
        independent_future_validation=False)
    signature=hashlib.sha256(
        json.dumps(contract,sort_keys=True).encode("utf-8")).hexdigest()
    return contract,signature


def load_policy_forecasts(root,policy):
    """Exact per-seed forecasts from a complete 06 policy only."""
    path=root/"policies"/(policy.id+".npz")
    with np.load(path,allow_pickle=False) as arrays:
        forecast=arrays["seed_predictions"].copy()
        mean=arrays["prediction"].copy()
    if (forecast.shape!=(10,12,59,3)
            or mean.shape!=(12,59,3)
            or not np.isfinite(forecast).all()
            or not np.allclose(mean,forecast.mean(0),atol=1e-12,rtol=0)):
        raise RuntimeError("Stage06 saved seed forecast bank invalid")
    return forecast


def run(args):
    """One fully registered 216-policy experiment; seed-wise resumable."""
    if args.dry_run:
        grid=enumerate_policies()
        print("06_CANONICAL_FULL_FACTORIAL",len(grid),"POLICIES")
        print("FACTORS",SLOTS,
              "LEVELS",len(MISSING),len(NORMALIZATION),len(DISTRIBUTION))
        print("RAW_FIT_UPPER_BOUND",len(grid)*len(SEEDS),
              "ACTUAL_LOWER_DUE_TO_BASELINE_AND_IDENTICAL_INPUTS")
        print("BASELINE",ORIGINAL.id,"SOURCE",RESULT05)
        print("TRAIN FULL 2006-2019, 36 COMMON ORIGINS 2011-2019")
        print("DEV 2020-2022, L1, B-MTGNN STRUCTURAL, 10 SEEDS, EPOCH 75")
        print("GLOBAL_OPTIMUM_WITHIN_ELIGIBLE_REGISTERED_GRID_ONLY")
        print("NO LEGACY SEQUENTIAL MODE; NO TRAIN CV; NO TEST")
        print("PLOTS design + 10 full-grid descriptive result plots")
        return None
    manifest_file=RESULTS/"run_manifest.json"
    if args.status:
        state=json_read(manifest_file) if manifest_file.is_file() else {}
        complete=len(list((RESULTS/"policies").glob("*.json")))
        seeds_done=len(list((RESULTS/"jobs").rglob("*.json")))
        print("06_FACTORIAL_STATUS",
              state.get("status","NOT_STARTED"),
              "COMPLETE_POLICIES",complete,"OF",GRID_SIZE,
              "TRAINED_SEED_JOBS",seeds_done,
              "NEXT",state.get("resume_job"),
              "SELECTED",state.get("selected_policy"),
              "TEST_OPENED",False)
        return state
    if args.design_plot:
        label=plot_design(RESULTS)
        print("06_FACTORIAL_DESIGN_ONLY",label,flush=True)
        return label
    if not args.run:
        raise ValueError("Use --dry-run, --status, --design-plot or --run")
    if args.max_minutes<=0:
        raise ValueError("Stage06 requires a positive per-call time budget")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    m,m5,panel,selection05,manifest05=verify_upstream_and_load()
    spec,signature=configuration(m,m5,selection05,manifest05,panel)
    old=json_read(manifest_file) if manifest_file.is_file() else None
    if old and (old.get("contract",{}).get("experiment")!=
                spec["experiment"] or old.get("signature")!=signature):
        raise RuntimeError(
            "Stage06 result folder must contain only THIS full-factorial "
            "source and upstream fingerprint; no legacy compatibility")
    if old and old.get("status")=="PASS":
        if len(list((RESULTS/"policies").glob("*.json")))!=GRID_SIZE:
            raise RuntimeError("Stage06 PASS without complete grid")
        return json_read(RESULTS/"selected_preprocessing.json")
    torch.set_num_threads(4)
    start=time.perf_counter()
    RESULTS.mkdir(parents=True,exist_ok=True)
    manifest=dict(
        schema="PRISM06_FULL_FACTORIAL_216",
        signature=signature,contract=spec,
        started_utc=(old or {}).get(
            "started_utc",datetime.now(timezone.utc).isoformat()),
        status="RUNNING",
        candidates_total=GRID_SIZE,candidates_complete=0,
        baseline_policy=ORIGINAL.id,
        model=MODEL,matched_seeds=list(SEEDS),
        selected_policy=None,
        test_opened=False)
    write_json(manifest_file,manifest)
    plot_design(RESULTS)
    prior=structural_prior(panel)
    original_stage05_parity(panel,prior)
    source_forecasts=original_stage05_seed_predictions(panel)
    source_means,source_individual=seed_policy_metrics(
        source_forecasts,panel)
    print("06_FULL_GRID_ORIGINAL_STAGE05_L1",source_means["axis_mae"],
          "ICI",source_means["ici_mae"],"GRID",GRID_SIZE,flush=True)
    grid=enumerate_policies()
    results={}
    hashes={}
    for idx,policy in enumerate(grid,1):
        row,paused=policy_summary(
            RESULTS,policy,panel,prior,device,signature,
            start,args.max_minutes,hashes)
        if paused is not None:
            manifest.update(
                status="PARTIAL_TIME_BUDGET",resume_job=paused,
                candidates_complete=len(results),
                distinct_model_inputs=len(hashes),
                duration_seconds=time.perf_counter()-start)
            write_json(manifest_file,manifest)
            print("06_FULL_GRID_RESUMABLE",len(results),"/",GRID_SIZE,
                  paused,flush=True)
            return manifest
        if row is None or row.get("status") not in ("PASS","NUMERICAL_FAILURE"):
            raise RuntimeError("Stage06 policy result lacks final status")
        results[policy.id]=row
        manifest.update(candidates_complete=len(results),
                        current_policy=policy.id)
        if idx==1 or idx%10==0 or idx==GRID_SIZE:
            write_json(manifest_file,manifest)
        if row.get("status")=="PASS":
            print("06_GRID_POLICY",idx,"/",GRID_SIZE,
                  policy.id,
                  "MAE",round(row["axis_mae"],6),
                  "ICI",round(row["ici_mae"],6),
                  "REUSED",row.get("identical_input_reused_from"),
                  flush=True)
        else:
            print("06_GRID_INFEASIBLE",idx,"/",GRID_SIZE,
                  policy.id,row.get("error"),flush=True)
    table=eligible_policy_table(grid,results)
    chosen=select_grid_policy(table)
    table["selected_policy"]=table.policy.eq(chosen.id)
    table.to_csv(RESULTS/"full_factorial_216_scores.csv",
                 index=False,encoding="utf-8-sig")
    figures=chart_full_factorial(RESULTS,table,results,chosen,panel)
    selected_forecasts=load_policy_forecasts(RESULTS,chosen)
    current_scores,current_individual=seed_policy_metrics(
        selected_forecasts,panel)
    reference_forecasts=load_policy_forecasts(RESULTS,ORIGINAL)
    if not np.array_equal(reference_forecasts,source_forecasts):
        raise RuntimeError("Stage05 L1 baseline drifted during exhaustive search")
    if any(abs(current_scores[k]-results[chosen.id][k])>1e-10
           for k in current_scores):
        raise RuntimeError("Stage06 selected ten-seed forecast/score mismatch")
    write_npz(RESULTS/"selected_vs_stage05_dev.npz",
              selected_seed_predictions=selected_forecasts,
              stage05_seed_predictions=reference_forecasts,
              mean_selected_prediction=selected_forecasts.mean(0),
              truth=panel.y[TRAIN_END:DEV_END],
              quarters=np.array(panel.quarters[TRAIN_END:DEV_END]))
    feasible_count=int(table.eligible.sum())
    numeric_fail=int(table.status.eq("NUMERICAL_FAILURE").sum())
    duplicates=int(table.identical_input_reused_from.notna().sum())
    selection=dict(
        schema="PRISM06_FULL_FACTORIAL_216",
        status="LOCKED",policy_id=chosen.id,
        preprocessing=asdict(chosen),
        grid_policy_count=GRID_SIZE,
        evaluated_policy_count=len(table),
        eligible_policy_count=feasible_count,
        numeric_infeasible_count=numeric_fail,
        identical_input_policy_aliases=duplicates,
        unique_eligible_grid_optimum=bool(
            table.loc[table.eligible,"axis_mae"].eq(
                float(current_scores["axis_mae"])).sum()==1),
        global_optimum_scope=(
            "LOWEST_CONSTRAINED_DEV_MAE_OF_ELIGIBLE_FEASIBLE_216_GRID"),
        baseline_policy=asdict(ORIGINAL),
        original_stage05_dev_metrics=source_means,
        selected_dev_metrics=current_scores,
        selected_seed_metrics=current_individual,
        original_stage05_seed_metrics=source_individual,
        model=MODEL,selected_upstream_lookback=LOOKBACK,
        train_through="2019Q4",
        dev="2020Q1-2022Q4",
        seed_count=len(SEEDS),fixed_epochs=75,
        no_train_internal_cv=True,test_opened=False,
        dev_reused_in_previous_stages=True,
        full_factorial_exploratory=True,
        not_independent_future_generalization=True,
        factor_transformation_order=[
            "distribution","missing","normalization"],
        selection_rule=spec["selection_rule"],
        constraint_rule=spec["eligibility_rule"],
        factorial_results="full_factorial_216_scores.csv",
        pairwise_interactions="factor_interactions.csv",
        pareto_frontier="mae_ici_pareto.csv",
        per_seed_scores="per_seed_policy_scores.csv",
        figures=figures,signature=signature,
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(RESULTS/"selected_preprocessing.json",selection)
    manifest.update(status="PASS",candidates_complete=GRID_SIZE,
                    selected_policy=chosen.id,
                    eligible_policy_count=feasible_count,
                    numerical_failures=numeric_fail,
                    aliases=duplicates,figures=figures,
                    elapsed_seconds=time.perf_counter()-start,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_file,manifest)
    print("06_FULL_FACTORIAL_PASS",chosen.id,
          "MAE",round(current_scores["axis_mae"],6),
          "ICI",round(current_scores["ici_mae"],6),
          "ELIGIBLE",feasible_count,"FAILED",numeric_fail,
          "ALIASES",duplicates,"PLOTS",len(figures),flush=True)
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--design-plot",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--max-minutes",type=float,default=25.)
    parser.add_argument("--device",choices=("cpu","cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    run(args)


if __name__=="__main__":
    main()
