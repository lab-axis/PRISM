#!/usr/bin/env python3
"""PRISM Stage07 — Train-only stable-correlation, greedy feature pruning.

MAIN EXPERIMENT: Stage06-selected B-MTGNN formulation, lookback and
preprocessing; 10 paired seeds, 75 epochs, all admissible Train origins,
12 Dev quarters and the existing fixed 20-slot core.
Remove redundant MAIN-INPUT features without changing the economic prior,
9D as-of score context, target anchor or the learned-graph mechanism.

0. Lock Stage04 -> Stage05 -> selected Stage06 policy. Inspect ONLY raw
   2006Q1–2019Q4 Train predictors for redundancy. 2020–2022 Dev never fits
   correlation, missing-data statistics, selection representatives or graph.
1. Compute pairwise Spearman on 66 local (42 target-history, 24 industry)
   and 91 truly global channels. NEVER treat 59 replicated global values
   in a quarter as 59 independent observations. Use quarter-block
   bootstraps, |rho| lower 10th percentile, and sign consistency.
2. Sort candidate features by Train observed fraction, stable correlation
   centrality, then name (fixed, deterministic rank). Keep a feature unless
   it is redundantly correlated with ALREADY RETAINED features of the SAME
   role (target-history / industry / global). Do not cluster, do not remove
   connected components as a group. Log the exact retained representative.
3. Evaluate thresholds None, .995, .990, .980, .950, .900. For every
   distinct candidate, zero only pruned input values and corresponding
   input missing indicators, preserving the 157-slot GNN parameter count.
   Train on the SAME full-Train refit origins, with Stage06's 75-epoch
   recipe; frozen Dev forecasts for 2020–22. The no-removal reference
   reuses all ten selected Stage06 seed predictions.
4. Every threshold is independently retrained as a genuinely compact
   K-input version, irrespective of masked screening performance. A
   compact candidate must improve 10-seed MEAN of individual three-axis
   MAEs, avoid mean seed-level ICI regression, and improve two Dev years. If
   none pass, preserve all 157 inputs. Masked results are diagnostic
   ONLY, not a gate that could incorrectly reject a compact winner.

These are CONDITIONAL main-input pruning experiments: Stage04's Structural
prior, graph, lagged target anchor and 9D as-of context deliberately remain
unchanged. Removing a target-history feature from main input therefore
does NOT mean eliminating all use of that economic signal, and this fact
is reported explicitly to prevent a misleading "fully removed" claim.

Command:
    py Sources/07_correlated_feature_removal.py --dry-run
    py Sources/07_correlated_feature_removal.py --analyze-only
    py -u Sources/07_correlated_feature_removal.py --run
    py Sources/07_correlated_feature_removal.py --status

These are descriptive repeated-Dev selection results, NOT independent Test
evidence or statistical significance. No legacy 07 cache is accepted.
Completed current jobs are resumable. No writes to Stages04/05/06 or 08.
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
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"Sources/07_correlated_feature_removal.py"
SOURCE06=ROOT/"Sources/06_preprocessing.py"
RESULT06=ROOT/"Results/06_preprocessing"
RESULTS=ROOT/"Results/07_correlated_feature_removal"

TRAIN_COUNT=56
DEV_COUNT=12
INDUSTRIES=59
FEATURE_COUNT=157
HISTORY_END=42
INDUSTRY_END=66
GLOBAL_COUNT=91
HORIZON=1
SEEDS=tuple(range(2026,2036))
THRESHOLDS=(None,.995,.990,.980,.950,.900)
BLOCK_QUARTERS=4
BOOTSTRAPS=40
BOOTSTRAP_SEED=20261013
LOWER_QUANTILE=.10
SIGN_STABILITY=.90
MIN_PAIR_QUARTERS=16
STAGE07_MIN_YEARS_BETTER=2
EPS=1e-9
FEATURE_ROLES=((0,HISTORY_END,"target_history"),
               (HISTORY_END,INDUSTRY_END,"industry_source"),
               (INDUSTRY_END,FEATURE_COUNT,"global_source"))
AXES=("profitability","productivity","growth")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".writing")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,
                               allow_nan=False),encoding="utf-8")
    os.replace(temp,path)


def atomic_npz(path:Path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".writing.npz")
    np.savez_compressed(temp,**arrays)
    os.replace(temp,path)


@lru_cache(maxsize=1)
def stage06():
    spec=importlib.util.spec_from_file_location("_prism07_fixed06",SOURCE06)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def role(index:int)->str:
    for first,last,label in FEATURE_ROLES:
        if first<=index<last:
            return label
    raise ValueError(f"Unknown feature index {index}")


def upstream():
    """Validate the canonical 04/05/06 locks and all 10 seed forecasts."""
    m06=stage06()
    m,m05,panel,selection05,manifest05=m06.verify_upstream_and_load()
    names=("selected_preprocessing.json","run_manifest.json")
    paths=[RESULT06/name for name in names]
    if not all(p.is_file() for p in paths):
        raise RuntimeError("Completed Stage06 selected policy required")
    lock,manifest=map(json_read,paths)
    contract,signature=m06.configuration(
        m,m05,selection05,manifest05,panel)
    if (lock.get("status")!="LOCKED"
            or manifest.get("status")!="PASS"
            or manifest.get("candidates_complete")!=216
            or lock.get("evaluated_policy_count")!=216
            or lock.get("policy_id")!=manifest.get("selected_policy")
            or lock.get("signature")!=signature
            or manifest.get("signature")!=signature
            or manifest.get("contract")!=contract
            or contract.get("source_sha256")!=sha(SOURCE06)
            or lock.get("model")!=m06.MODEL
            or lock.get("seed_count")!=len(SEEDS)
            or tuple(contract["seed_grid"])!=SEEDS
            or lock.get("selected_upstream_lookback")!=m06.selected_lookback()
            or contract.get("lookback")!=m06.selected_lookback()
            or contract.get("fixed_epochs")!=75
            or m06.MODEL!="B-MTGNN"
            or contract.get("output_mode")!=m06.selected_mode()
            or lock.get("test_opened") is not False):
        raise RuntimeError("Stage06 selected input/model/factorial lock changed")
    policy=m06.Policy(**lock["preprocessing"])
    policy.validate()
    if policy.id!=lock["policy_id"]:
        raise RuntimeError("Stage06 preprocessing fields/identifier disagree")
    prior=m06.structural_prior(panel)
    folded,_=m06.prepare_policy(panel,policy,prior)
    lookback=m06.selected_lookback()
    mode=m06.selected_mode()
    if (folded["train_x"].shape!=(len(m06.train_origins()),lookback,INDUSTRIES,FEATURE_COUNT)
            or folded["val_x"].shape!=(DEV_COUNT,lookback,INDUSTRIES,FEATURE_COUNT)):
        raise RuntimeError("Stage06 selected model input contract differs")
    bank=m06.load_policy_forecasts(RESULT06,policy)
    with np.load(RESULT06/"selected_vs_stage05_dev.npz",
                 allow_pickle=False) as z:
        if (not np.array_equal(bank,z["selected_seed_predictions"])
                or not np.array_equal(z["truth"],panel.y[TRAIN_COUNT:])):
            raise RuntimeError("Stage06 frozen seed forecasts drifted")
    baseline,individual=m06.seed_policy_metrics(bank,panel)
    if any(abs(baseline[key]-float(expected))>1e-10
           for key,expected in lock["selected_dev_metrics"].items()):
        raise RuntimeError("Stage06 locked 10-seed Dev scores drifted")
    model=m05.instantiate(
        m,"B-MTGNN",SEEDS[0],lookback,folded,mode)
    full_params=sum(p.numel() for p in model.parameters())
    return dict(
        m04=m,m05=m05,panel=panel,m06=m06,
        models=("B-MTGNN",),epochs=m.STAGE04_ACTIVE_EPOCHS,
        lookback=lookback,output_mode=mode,
        locked_policy=policy,prior=prior,full_fold=folded,
        original_forecasts=bank,baseline_metrics=baseline,
        baseline_seed_metrics=individual,baseline_parameters=full_params,
        upstream06_lock_sha256=sha(paths[0]),
        upstream06_manifest_sha256=sha(paths[1]),
        stage04_sha256=sha(m06.SOURCE04),
        panel_hashes=panel.hashes,
        upstream06_input_sha256=m06.input_fingerprint(folded))


def spearman(data:np.ndarray,min_periods:int)->np.ndarray:
    """Pairwise-complete rank correlation. No target-label relevance."""
    if data.ndim!=2:
        raise ValueError("Expected an observation-by-feature matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        matrix=pd.DataFrame(data).corr(
            method="spearman",min_periods=min_periods).to_numpy(float)
    return np.nan_to_num(matrix,nan=0.,posinf=0.,neginf=0.)


def correlation_views(raw_train:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """[59-node local] and [one-quarter global] distinct sampling units.

    Local/local: the 59×T actual industry observations, with >=16*12
    observed industry-quarter pairs for a usable Spearman coefficient.
    Global/global: exactly T rows (one per quarter, no 59-fold duplicate).
    Local/global: quarterly median of local feature over industries
    compared with the single quarterly global series.
    Support is independent QUARTER co-presence, not an iid sample count.
    """
    raw=np.asarray(raw_train,np.float64)
    if (raw.ndim!=3 or raw.shape[1:]!=(INDUSTRIES,FEATURE_COUNT)
            or len(raw)<20 or np.isinf(raw).any()):
        raise ValueError("Correlation uses raw [Train quarters,59,157] only")
    node=raw[:,:,:INDUSTRY_END]
    global_data=raw[:,:,INDUSTRY_END:]
    baseline=global_data[:,:1]
    if not np.allclose(global_data,baseline,equal_nan=True,
                       rtol=1e-10,atol=1e-10):
        raise RuntimeError("Global economic channels are not constant over nodes")
    unique_global=baseline[:,0,:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        quarterly_node=np.nanmedian(node,axis=1)
    nn=spearman(
        node.reshape(-1,INDUSTRY_END),MIN_PAIR_QUARTERS*12)
    gg=spearman(unique_global,MIN_PAIR_QUARTERS)
    ng=spearman(
        np.column_stack((quarterly_node,unique_global)),
        MIN_PAIR_QUARTERS)[:INDUSTRY_END,INDUSTRY_END:]
    rho=np.eye(FEATURE_COUNT,dtype=np.float64)
    rho[:INDUSTRY_END,:INDUSTRY_END]=nn
    rho[INDUSTRY_END:,INDUSTRY_END:]=gg
    rho[:INDUSTRY_END,INDUSTRY_END:]=ng
    rho[INDUSTRY_END:,:INDUSTRY_END]=ng.T
    rho=np.clip((rho+rho.T)/2,-1.,1.)
    np.fill_diagonal(rho,1.)
    available=np.column_stack((
        np.isfinite(node).any(axis=1),np.isfinite(unique_global)))
    support=(available.astype(np.int32).T @ available.astype(np.int32))
    return rho,support


def moving_block_indices(quarters:int,rng:np.random.Generator,
                         width:int=BLOCK_QUARTERS)->np.ndarray:
    if width<1 or quarters<width:
        raise ValueError("Insufficient quarters for causal block resampling")
    starts=rng.integers(
        0,quarters-width+1,size=int(np.ceil(quarters/width)))
    return np.concatenate([np.arange(v,v+width) for v in starts])[:quarters]


@dataclass
class CorrelationAudit:
    rho:np.ndarray
    support:np.ndarray
    lower_abs:np.ndarray
    sign_consistency:np.ndarray
    observed_fraction:np.ndarray
    draws:int


def assess_train_correlations(raw_train:np.ndarray,
                              draws:int=BOOTSTRAPS,
                              seed:int=BOOTSTRAP_SEED,
                              width:int=BLOCK_QUARTERS)->CorrelationAudit:
    if draws<2:
        raise ValueError("Need repeated quarter-block draws")
    rho,support=correlation_views(raw_train)
    rng=np.random.default_rng(seed)
    samples=np.empty((draws,FEATURE_COUNT,FEATURE_COUNT),np.float32)
    for i in range(draws):
        quarter_indices=moving_block_indices(
            len(raw_train),rng,width=width)
        samples[i]=correlation_views(raw_train[quarter_indices])[0]
    lower=np.quantile(np.abs(samples),LOWER_QUANTILE,axis=0)
    sign=(np.sign(samples)==np.sign(rho)[None]).mean(axis=0)
    supported=support>=MIN_PAIR_QUARTERS
    lower=np.where(supported,lower,0.)
    sign=np.where(supported,sign,0.)
    np.fill_diagonal(lower,1.)
    np.fill_diagonal(sign,1.)
    availability=np.isfinite(raw_train).mean(axis=(0,1))
    return CorrelationAudit(rho,support,lower,sign,availability,draws)


def stable_edges(audit:CorrelationAudit,threshold:float)->np.ndarray:
    if threshold not in THRESHOLDS[1:]:
        raise ValueError("Unknown Stage07 predeclared Spearman threshold")
    edges=(
        (np.abs(audit.rho)>=threshold)
        & (audit.lower_abs>=threshold)
        & (audit.sign_consistency>=SIGN_STABILITY)
        & (audit.support>=MIN_PAIR_QUARTERS))
    # Guard type-specific meaning: do NOT prune a macro variable because
    # an industrial Target history happens to move with it.
    same_role=np.zeros_like(edges,dtype=bool)
    for first,last,_ in FEATURE_ROLES:
        same_role[first:last,first:last]=True
    edges &=same_role
    edges=np.logical_and(edges,edges.T)
    np.fill_diagonal(edges,False)
    return edges


@dataclass
class Subset:
    label:str
    threshold:float|None
    kept:tuple[int,...]
    dropped:tuple[int,...]
    mapping:pd.DataFrame
    stable_edge_count:int


def greedy_prune(audit:CorrelationAudit,features:list[str],
                 threshold:float|None)->Subset:
    """Fixed-order greedy independent set; no hierarchical clustering.

    A→B and B→C high correlation never imply A→C redundant. A rejected
    feature MUST have a direct stable high-correlation edge to an
    ALREADY RETAINED feature from the SAME semantic role.
    """
    n=len(features)
    if n not in (3,FEATURE_COUNT):
        raise ValueError("Unexpected test/full feature dimensions")
    if len(set(features))!=n:
        raise ValueError("Duplicate feature names")
    if (audit.rho.shape!=(n,n)
            or audit.lower_abs.shape!=(n,n)
            or len(audit.observed_fraction)!=n):
        raise ValueError("Correlation audit dimension mismatch")
    label=("no_removal" if threshold is None else
           f"greedy_{threshold:.3f}")
    if threshold is None:
        edges=np.zeros((n,n),bool)
    else:
        if n==FEATURE_COUNT:
            edges=stable_edges(audit,threshold)
        else:
            # Small synthetic graph test uses same first role 0..2.
            edges=((np.abs(audit.rho)>=threshold)
                   &(audit.lower_abs>=threshold)
                   &(audit.sign_consistency>=SIGN_STABILITY)
                   &(audit.support>=MIN_PAIR_QUARTERS))
            np.fill_diagonal(edges,False)
    stable_base=(
        (np.abs(audit.rho)>=min(THRESHOLDS[1:]))
        & (audit.lower_abs>=min(THRESHOLDS[1:]))
        & (audit.sign_consistency>=SIGN_STABILITY)
        & (audit.support>=MIN_PAIR_QUARTERS))
    np.fill_diagonal(stable_base,False)
    # Centrality ranking is calculated ONCE independently of threshold.
    centrality=(np.abs(audit.rho)*stable_base).sum(axis=1)
    priority=sorted(range(n),key=lambda i:(
        -float(audit.observed_fraction[i]),
        -float(centrality[i]),
        features[i]))
    kept=[]
    representations={}
    for i in priority:
        redundant=[j for j in kept if edges[i,j]]
        if redundant:
            # Highest reliability first; deterministic feature-name tie.
            representative=min(redundant,key=lambda j:(
                -float(audit.lower_abs[i,j]),
                -float(abs(audit.rho[i,j])),features[j]))
            representations[i]=representative
        else:
            kept.append(i)
            representations[i]=i
    kept=tuple(sorted(kept))
    drop=tuple(sorted(set(range(n))-set(kept)))
    records=[]
    for i in range(n):
        j=representations[i]
        records.append(dict(
            threshold=label,feature=features[i],feature_index=i,
            role=role(i),retained=(i==j),
            representative=features[j],representative_index=j,
            train_observed_fraction=float(audit.observed_fraction[i]),
            stable_rho=float(audit.rho[i,j]),
            absolute_spearman=float(abs(audit.rho[i,j])),
            bootstrap_q10_abs=float(audit.lower_abs[i,j]),
            bootstrap_sign_agreement=float(audit.sign_consistency[i,j]),
            unique_train_quarters_support=int(audit.support[i,j])))
    table=pd.DataFrame(records)
    if (len(kept)+len(drop)!=n
            or any(not edges[i,representations[i]] for i in drop)):
        raise RuntimeError("Greedy pruning removed an unsupported feature")
    return Subset(label,threshold,kept,drop,table,
                  int(np.triu(edges,1).sum()))


def adapt_inputs(full:dict,subset:Subset,mode:str)->dict:
    """Do not change targets, economic prior, lag anchor, graph or context.

    masked: SAME input dimensionality and parameter count as 04; set
      removed channels of BOTH numerical values and missing-mask to zero.
    compact: real K-dimensional input layer; slice those four input tensors
      and the two feature-scaler buffers, nothing else.

    This studies main-input redundancy conditional on the explicitly
    retained fixed Stage04 context/anchor/prior, NOT full causal removal
    of a signal from every path.
    """
    if mode not in ("masked","compact"):
        raise ValueError(mode)
    indices=np.asarray(subset.kept,dtype=int)
    if (not len(indices) or not np.array_equal(indices,np.unique(indices))
            or indices[0]<0 or indices[-1]>=FEATURE_COUNT):
        raise ValueError("Invalid selected predictor indices")
    changed={}
    channel_keys=("train_x","train_mask","val_x","val_mask")
    for key,v in full.items():
        if isinstance(v,np.ndarray):
            if key in channel_keys:
                if mode=="masked":
                    data=np.array(v,copy=True)
                    data[...,list(subset.dropped)]=0.
                else:
                    data=np.ascontiguousarray(v[...,indices])
            elif key in ("x_center","x_scale") and mode=="compact":
                data=np.ascontiguousarray(v[indices])
            else:
                data=np.array(v,copy=True)
            changed[key]=data
        else:
            changed[key]=v
    channels=(FEATURE_COUNT if mode=="masked" else len(indices))
    lookback=full["train_x"].shape[1]
    if (changed["train_x"].shape!=(len(full["train_origins"]),lookback,INDUSTRIES,channels)
            or changed["val_x"].shape!=(12,lookback,INDUSTRIES,channels)):
        raise RuntimeError("Changed the Stage06 selected model-input shapes")
    for key in ("train_y","val_y","train_last","val_last",
                "train_context","val_context",
                "train_structure","val_structure",
                "y_center","y_scale","train_origins","val_origins","anchor_biases"):
        if not np.array_equal(changed[key],full[key]):
            raise RuntimeError("Subset pruning silently changed a non-input path")
    if mode=="masked":
        for key in channel_keys:
            if np.any(changed[key][...,list(subset.dropped)]!=0):
                raise RuntimeError("Masked feature still reaches input projection")
    return changed


def dataset_sha(data:dict)->str:
    hasher=hashlib.sha256()
    for key in ("train_x","train_mask","train_y","train_last",
                "train_context","train_structure",
                "val_x","val_mask","val_y","val_last",
                "val_context","val_structure","x_center","x_scale",
                "y_center","y_scale","train_origins","val_origins","anchor_biases"):
        array=np.ascontiguousarray(data[key])
        hasher.update(key.encode()+b"\0"+str(array.shape).encode()+b"\0")
        hasher.update(str(array.dtype).encode()+array.tobytes())
    return hasher.hexdigest()


def selected_baseline_scores(ctx):
    return dict(ctx["baseline_metrics"])


def selection_safeguards(candidate:dict,baseline:dict)->dict:
    """Same conservative development safeguards as Stage06 (not a Test)."""
    year_wins=sum(
        candidate[f"{year}_mae"]<
        baseline[f"{year}_mae"]-EPS
        for year in (2020,2021,2022))
    improvement=(
        candidate["axis_mae"]<baseline["axis_mae"]-EPS)
    ici_safe=(
        candidate["ici_mae"]<=baseline["ici_mae"]+EPS)
    return dict(
        three_axis_improvement=bool(improvement),
        ici_nonworsening=bool(ici_safe),
        years_improved=int(year_wins),
        qualified=bool(improvement and ici_safe
                       and year_wins>=STAGE07_MIN_YEARS_BETTER))


def select_compact_candidate(subsets,compact_rows,baseline):
    """Select from REAL K-input model predictions, not masked proxies."""
    if len(compact_rows)!=len(subsets)-1:
        raise ValueError("All five compact candidates must complete")
    by_label={row["policy"]:row for row in compact_rows}
    if len(by_label)!=len(compact_rows):
        raise ValueError("Duplicate compact candidate")
    valid=[]
    for subset in subsets[1:]:
        record=by_label.get(subset.label)
        if record is None or record["kept_count"]!=len(subset.kept):
            raise RuntimeError("Incomplete compact candidate contract")
        if selection_safeguards(record,baseline)["qualified"]:
            valid.append((subset,record))
    if not valid:
        return subsets[0],baseline
    winner,score=min(valid,key=lambda pair:(
        pair[1]["axis_mae"],-len(pair[0].dropped),pair[0].label))
    return winner,score


def _candidate_folder(root:Path,subset:Subset,mode:str):
    return root/"models"/f"{subset.label}__{mode}"


def instantiate_subset(ctx,subset:Subset,seed:int,compact:bool):
    """Share all retained initial weights with the seeded 157-input model."""
    m=ctx["m04"]
    m05=ctx["m05"]
    net=m05.instantiate(
        m,"B-MTGNN",seed,ctx["lookback"],ctx["full_fold"],ctx["output_mode"])
    k=len(subset.kept) if compact else FEATURE_COUNT
    if k!=FEATURE_COUNT:
        if k!=len(subset.kept):
            raise RuntimeError("Compact input dimension disagrees with subset")
        old_values=net.base.values
        old_missing=net.base.missing
        rng=torch.get_rng_state()
        values=torch.nn.Linear(k,old_values.out_features)
        missing=torch.nn.Linear(
            k,old_missing.out_features,bias=False)
        with torch.no_grad():
            values.weight.copy_(old_values.weight[:,list(subset.kept)])
            values.bias.copy_(old_values.bias)
            missing.weight.copy_(old_missing.weight[:,list(subset.kept)])
        net.base.values=values
        net.base.missing=missing
        torch.set_rng_state(rng)
    parameters=sum(p.numel() for p in net.parameters())
    expected=(ctx["baseline_parameters"]-
              2*m.STAGE04_ACTIVE_WIDTH*(FEATURE_COUNT-k))
    if parameters!=expected:
        raise RuntimeError("Unexpected B-MTGNN compact parameter count")
    return net


def fit_subset_model(ctx,subset:Subset,data:dict,seed:int,device):
    """Stage06 75-epoch Train recipe with Stage05's fixed 20-slot core."""
    m=ctx["m04"]
    k=data["train_x"].shape[-1]
    if k not in (FEATURE_COUNT,len(subset.kept)):
        raise RuntimeError("Subset dataset/model channels disagree")
    net=instantiate_subset(ctx,subset,seed,compact=k!=FEATURE_COUNT).to(device)
    parameters=sum(p.numel() for p in net.parameters())
    z={key:torch.as_tensor(data[key],device=device)
       for key in ("train_x","train_mask","train_y","train_last",
                   "train_context","train_structure","val_x","val_mask",
                   "val_last","val_context","val_structure")}
    graph=torch.eye(m.NODE_COUNT,device=device)
    scale=torch.as_tensor(data["y_scale"],device=device)
    optim=torch.optim.AdamW(
        net.parameters(),lr=m.FIXED_LEARNING_RATE,weight_decay=1e-4)
    started=time.perf_counter()
    for _ in range(m.STAGE04_ACTIVE_EPOCHS):
        net.train()
        optim.zero_grad(set_to_none=True)
        output=net(z["train_x"],z["train_mask"],graph,
                   z["train_last"],z["train_structure"],z["train_context"])
        loss=m.joint_axis_ici_loss(output,z["train_y"],scale)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite Stage07 Train loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),5.,error_if_nonfinite=True)
        optim.step()
    with m.isolated_snapshot_inference(net):
        prediction=m.mtgnn_bayesian_inference(
            net,z,graph,"B-MTGNN",seed,
            data["y_center"],data["y_scale"])["prediction"]
    state={key:value.detach().cpu().clone()
           for key,value in net.state_dict().items()}
    seconds=float(time.perf_counter()-started)
    del net,optim,z
    if device.type=="cuda":
        torch.cuda.empty_cache()
    if prediction.shape!=(DEV_COUNT,INDUSTRIES,3) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid Stage07 10-seed member forecast")
    return dict(
        prediction=prediction,state_dict=state,parameters=parameters,
        seconds=seconds,inference_draws=m.STAGE04_BMTGNN_MC_SAMPLES,
        inference_regime="B_MTGNN_MC_DROPOUT")


def run_subset(ctx,root,subset:Subset,mode:str,signature:str,device):
    """Fit or resume exactly ten matched seed jobs; score seeds separately."""
    if mode=="compact" and not subset.dropped:
        raise ValueError("Nothing to compact")
    if not subset.dropped:
        return dict(
            **selected_baseline_scores(ctx),
            parameters=ctx["baseline_parameters"],
            model_count=1,seed_count=len(SEEDS),
            feature_count=FEATURE_COUNT,removed_count=0,
            reused_upstream=True)
    changed=adapt_inputs(ctx["full_fold"],subset,mode)
    provenance=dataset_sha(changed)
    folder=_candidate_folder(root,subset,mode)
    panel=ctx["panel"]
    predictions=[]
    parameters=(ctx["baseline_parameters"]-
                2*ctx["m04"].STAGE04_ACTIVE_WIDTH*len(subset.dropped)
                if mode=="compact" else ctx["baseline_parameters"])
    for seed in SEEDS:
        stem=f"B-MTGNN__s{seed}"
        meta=folder/(stem+".json")
        npz=folder/(stem+".npz")
        ck=folder/(stem+".pt")
        if meta.is_file() and npz.is_file() and ck.is_file():
            saved=json_read(meta)
            if (saved.get("contract_sha256")!=signature
                    or saved.get("input_sha256")!=provenance
                    or saved.get("model")!="B-MTGNN"
                    or saved.get("seed")!=seed
                    or saved.get("epoch")!=ctx["epochs"]
                    or saved.get("parameters")!=parameters
                    or saved.get("variant")!=mode
                    or saved.get("subset")!=subset.label
                    or saved.get("prediction_sha256")!=sha(npz)
                    or saved.get("checkpoint_sha256")!=sha(ck)):
                raise RuntimeError("Stage07 cache from a different run")
            with np.load(npz,allow_pickle=False) as z:
                result=z["prediction"].copy()
            if result.shape!=(12,59,3) or not np.isfinite(result).all():
                raise RuntimeError("Damaged Stage07 prediction")
            chk=torch.load(ck,map_location="cpu",weights_only=True)
            if (chk["contract_sha256"]!=signature
                    or chk["model"]!="B-MTGNN" or chk["seed"]!=seed
                    or chk["parameters"]!=parameters
                    or chk["feature_count"]!=changed["train_x"].shape[-1]
                    or chk["input_sha256"]!=provenance):
                raise RuntimeError("Checkpoint does not match cached prediction")
        elif meta.exists() or npz.exists() or ck.exists():
            raise RuntimeError(f"Incomplete Stage07 checkpoint {folder}/{stem}")
        else:
            fitted=fit_subset_model(ctx,subset,changed,seed,device)
            result=fitted["prediction"]
            folder.mkdir(parents=True,exist_ok=True)
            checkpoint=dict(
                contract_sha256=signature,subset=subset.label,
                variant=mode,model="B-MTGNN",seed=seed,
                feature_count=changed["train_x"].shape[-1],
                parameters=parameters,input_sha256=provenance,
                state_dict=fitted["state_dict"])
            temp=ck.with_suffix(".pt.writing")
            torch.save(checkpoint,temp)
            torch.load(temp,map_location="cpu",weights_only=True)
            os.replace(temp,ck)
            atomic_npz(npz,prediction=result)
            atomic_json(meta,dict(
                contract_sha256=signature,input_sha256=provenance,
                prediction_sha256=sha(npz),checkpoint_sha256=sha(ck),
                subset=subset.label,variant=mode,
                model="B-MTGNN",seed=seed,epoch=ctx["epochs"],
                training_origins=changed["train_origins"].tolist(),
                parameters=parameters,fit_seconds=float(fitted["seconds"]),
                inference_draws=int(fitted["inference_draws"]),
                inference_regime=fitted["inference_regime"],
                test_opened=False))
            print("07_FIT",subset.label,mode,
                  "B-MTGNN",seed,"K",changed["train_x"].shape[-1],
                  "EPOCH",ctx["epochs"],flush=True)
        predictions.append(result)
    scores,individual=ctx["m06"].seed_policy_metrics(
        np.stack(predictions),panel)
    scores.update(
        feature_count=(FEATURE_COUNT if mode=="masked" else len(subset.kept)),
        removed_count=len(subset.dropped),
        parameters=parameters,
        model_count=1,
        seed_count=len(SEEDS),
        reused_upstream=False,seed_metrics=individual)
    return scores


def run_contract(ctx)->dict:
    """Run signature blocks mixed source/data/model/correlation caches."""
    return dict(
        stage="07_CORRELATION_GREEDY_NO_CLUSTERING",
        source_sha256=sha(SOURCE),
        stage04_source_sha256=ctx["stage04_sha256"],
        stage06_source_sha256=sha(SOURCE06),
        stage06_lock_sha256=ctx["upstream06_lock_sha256"],
        stage06_manifest_sha256=ctx["upstream06_manifest_sha256"],
        stage03_panel_hashes=ctx["panel_hashes"],
        stage06_model_input_sha256=ctx["upstream06_input_sha256"],
        stage06_policy=ctx["locked_policy"].id,
        selected_models=["B-MTGNN"],
        seeds=list(SEEDS),
        epochs=ctx["epochs"],
        baseline_parameters=ctx["baseline_parameters"],
        feature_count=FEATURE_COUNT,industries=INDUSTRIES,
        train_quarters=TRAIN_COUNT,dev_quarters=DEV_COUNT,
        train_origins=ctx["full_fold"]["train_origins"].tolist(),
        lookback=ctx["lookback"],forecast_horizon=HORIZON,
        fixed_temporal_core=ctx["m05"].CORE_LOOKBACK,
        mc_dropout_draws=ctx["m04"].STAGE04_BMTGNN_MC_SAMPLES,
        output_mode=ctx["output_mode"],
        thresholds=list(THRESHOLDS),
        bootstraps=BOOTSTRAPS,
        bootstrap_quarters=BLOCK_QUARTERS,
        bootstrap_seed=BOOTSTRAP_SEED,
        descriptive_bootstrap_sensitivity=[
            [2,40],[4,20],[4,40],[4,80],[8,40]],
        min_support_unique_train_quarters=MIN_PAIR_QUARTERS,
        bootstrap_abs_q10=LOWER_QUANTILE,
        stable_sign_fraction=SIGN_STABILITY,
        semantic_groups=[role for _,_,role in FEATURE_ROLES],
        cluster_algorithms_used=False,
        pruning_method="fixed_order_greedy_direct_to_kept",
        validation_metric="TEN_INDIVIDUAL_SEED_METRICS_THEN_MEAN",
        selection_safeguards="MAE better, ICI nonworse, >=2 of 3 years",
        first_pass_157_dim_masked=True,
        primary_selection_all_five_real_K_dim_compact=True,
        masked_results_diagnostic_not_compact_gate=True,
        compact_initialization="ORIGINAL_157_SEEDED_WEIGHTS_SUBSET_COLUMNS",
        unchanged_context_prior_learned_graph_anchor=True,
        dev_reused_from_stages_04_05_06=True,
        independent_future_validation=False,
        test_opened=False)


def run_signature(contract:dict)->str:
    return hashlib.sha256(
        json.dumps(contract,sort_keys=True,
                   ensure_ascii=False).encode()).hexdigest()


def cached_audit(ctx,root:Path,signature:str):
    archive=root/"correlations"/"train_spearman_bootstrap.npz"
    meta=root/"correlations"/"provenance.json"
    if archive.exists() != meta.exists():
        raise RuntimeError("Incomplete correlation audit cache")
    if archive.exists():
        state=json_read(meta)
        if state.get("signature")!=signature:
            raise RuntimeError("Correlation cache from changed study/Train")
        with np.load(archive,allow_pickle=False) as z:
            audit=CorrelationAudit(
                rho=z["rho"].copy(),support=z["support"].copy(),
                lower_abs=z["lower_abs"].copy(),
                sign_consistency=z["sign_consistency"].copy(),
                observed_fraction=z["observed_fraction"].copy(),
                draws=int(z["draws"]))
        return audit
    audit=assess_train_correlations(ctx["panel"].x[:TRAIN_COUNT])
    atomic_npz(
        archive,rho=audit.rho,support=audit.support,
        lower_abs=audit.lower_abs,
        sign_consistency=audit.sign_consistency,
        observed_fraction=audit.observed_fraction,
        draws=np.asarray(audit.draws))
    atomic_json(meta,dict(
        signature=signature,
        source="Stage03 ORIGINAL raw_train predictors only",
        processed_train_through="2019Q4",
        dev_or_test_distribution_used=False,
        raw_global_sources_duplicated_over_industries_not_counted_as_independent=True,
        bootstrap_draws=audit.draws,
        random_seed=BOOTSTRAP_SEED,
        semantic_groups_separate=True))
    return audit


def candidate_subsets(audit,ctx,root):
    names=ctx["panel"].features
    subsets=tuple(greedy_prune(audit,names,t) for t in THRESHOLDS)
    root.mkdir(parents=True,exist_ok=True)
    counts=[]
    for subset in subsets:
        subset.mapping.to_csv(
            root/(subset.label+".csv"),index=False,encoding="utf-8-sig")
        counts.append(dict(
            threshold="none" if subset.threshold is None else subset.threshold,
            label=subset.label,
            total_features=FEATURE_COUNT,
            kept_count=len(subset.kept),
            dropped_count=len(subset.dropped),
            strong_within_role_edge_count=subset.stable_edge_count,
            target_history_kept=sum(i<HISTORY_END for i in subset.kept),
            industry_source_kept=sum(HISTORY_END<=i<INDUSTRY_END for i in subset.kept),
            global_source_kept=sum(i>=INDUSTRY_END for i in subset.kept)))
    pd.DataFrame(counts).to_csv(
        root/"candidate_counts.csv",index=False,encoding="utf-8-sig")
    return subsets


def save_bootstrap_sensitivity(ctx,root,audit,subsets):
    """Train-only sensitivity of the chosen greedy subsets, not a new gate."""
    raw=ctx["panel"].x[:TRAIN_COUNT]
    names=ctx["panel"].features
    records=[]
    for width,draws in ((2,40),(4,20),(4,40),(4,80),(8,40)):
        alternate=(audit if (width,draws)==(BLOCK_QUARTERS,BOOTSTRAPS)
                   else assess_train_correlations(
                       raw,draws=draws,seed=BOOTSTRAP_SEED,width=width))
        for reference in subsets[1:]:
            candidate=greedy_prune(
                alternate,names,reference.threshold)
            original=set(reference.kept)
            current=set(candidate.kept)
            records.append(dict(
                block_quarters=width,bootstrap_draws=draws,
                threshold=reference.threshold,
                reference_kept=len(original),
                sensitivity_kept=len(current),
                different_feature_decisions=len(original^current),
                retained_jaccard=len(original&current)/len(original|current)))
    target=root/"correlations"/"block_sensitivity.csv"
    target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(
        target,index=False,encoding="utf-8-sig")
    return str(target.relative_to(root)).replace("\\","/")


def plots(root,ctx,audit,subsets,comparisons,compact_comparisons,
          chosen:Subset,final_mode:str,final_metrics:dict):
    """Six descriptive plots for masked and real compact candidates."""
    dest=root/"plots"
    dest.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(comparisons)
    compact=pd.DataFrame(compact_comparisons).set_index("policy")
    ordered=[subset.label for subset in subsets]
    df=df.set_index("policy").loc[ordered].reset_index()
    tags=["none" if s.threshold is None else f"{s.threshold:.3f}"
          for s in subsets]
    figures=[]
    def save(fig,name):
        fig.tight_layout()
        target=dest/(name+".png")
        fig.savefig(target,dpi=170,bbox_inches="tight")
        plt.close(fig)
        figures.append("plots/"+target.name)

    fig,ax=plt.subplots(figsize=(11.6,5.5))
    ax.plot(df.kept_count,df.axis_mae,"o--",lw=1.2,
            label="Masked 157-input diagnostic")
    ax.plot(compact.kept_count,compact.axis_mae,"s-",lw=1.7,
            label="Actual compact K-input GNNs")
    for k,r in enumerate(df.itertuples(index=False)):
        ax.annotate(tags[k],(r.kept_count,r.axis_mae),
                    xytext=(2,6),textcoords="offset points",fontsize=8)
    ax.axhline(float(df.iloc[0].axis_mae),ls="--",lw=1,
               label="Frozen 10-seed Stage06 baseline")
    ax.set(xlabel="Number of main-input features kept",
           ylabel="2020–2022 mean of seed-wise Dev 3-axis MAEs",
           title="Masked diagnostic vs compact B-MTGNN feature removal")
    ax.grid(alpha=.3);ax.legend()
    save(fig,"07_01_kept_features_vs_dev_mae")

    fig,ax=plt.subplots(figsize=(11.5,5.1))
    xx=np.arange(len(subsets))
    count=pd.read_csv(root/"subsets"/"candidate_counts.csv")
    bottom=np.zeros(len(subsets))
    for column,title in (
        ("target_history_kept","Target history"),
        ("industry_source_kept","Industry sources"),
        ("global_source_kept","Global economic sources")):
        y=count[column].to_numpy()
        ax.bar(xx,y,bottom=bottom,label=title)
        bottom+=y
    ax.set_xticks(xx,tags)
    ax.set(xlabel="Stable absolute Spearman threshold",
           ylabel="Retained main-input features",
           title="Only SAME-ROLE redundant pairs are eligible to prune")
    ax.legend();ax.grid(axis="y",alpha=.2)
    save(fig,"07_02_features_by_role")

    fig,ax=plt.subplots(figsize=(11.8,5.2))
    for year in (2020,2021,2022):
        ax.plot(xx,df[f"{year}_mae"]-df.iloc[0][f"{year}_mae"],
                "o--",alpha=.45,label=f"{year} masked diagnostic")
        compact_values=[float(compact.loc[s.label,f"{year}_mae"])
                        for s in subsets[1:]]
        ax.plot(xx[1:],np.asarray(compact_values)-
                float(df.iloc[0][f"{year}_mae"]),
                "s-",label=f"{year} actual compact")
    ax.axhline(0,color="black",lw=1)
    ax.set_xticks(xx,tags)
    ax.set(xlabel="Greedy redundancy threshold",
           ylabel="Dev annual MAE minus no-removal (negative = better)",
           title="Year-level robustness: no independent economic observations invented")
    ax.legend();ax.grid(alpha=.25)
    save(fig,"07_03_each_dev_year")

    fig,ax=plt.subplots(figsize=(11.2,5.0))
    cols=[a+"_mae" for a in AXES]+["ici_mae"]
    base=[float(df.iloc[0][c]) for c in cols]
    end=[float(final_metrics[c]) for c in cols]
    x=np.arange(len(cols))
    ax.bar(x-.2,base,width=.4,label="Original 157 inputs")
    ax.bar(x+.2,end,width=.4,label=(
        f"Adopted {len(chosen.kept)} inputs ({final_mode})"))
    ax.set_xticks(x,[*AXES,"ICI"])
    ax.set(ylabel="Dev mean absolute error",
           title="Unchanged three targets, Structural prior, and context")
    ax.grid(axis="y",alpha=.2);ax.legend()
    save(fig,"07_04_three_axes_ici")

    fig,ax=plt.subplots(figsize=(10.4,8.4))
    z=np.abs(audit.rho)
    picture=ax.imshow(z,vmin=0,vmax=1,aspect="auto",cmap="viridis")
    for start in (HISTORY_END-.5,INDUSTRY_END-.5):
        ax.axvline(start,color="white",lw=1)
        ax.axhline(start,color="white",lw=1)
    fig.colorbar(picture,ax=ax,label="Absolute raw Train Spearman rho")
    ax.set(title="TRAIN-only correlations (not clustered)",
           xlabel="Original ordered features",ylabel="Original ordered features")
    save(fig,"07_05_train_spearman_overview")

    fig,ax=plt.subplots(figsize=(8.5,6.2))
    pareto=pd.read_csv(root/"compact_pareto.csv")
    ax.scatter(pareto.axis_mae,pareto.ici_mae,c=pareto.kept_count,
               s=90,cmap="viridis",edgecolor="black",linewidth=.5)
    for row in pareto.itertuples(index=False):
        ax.annotate(row.policy.replace("greedy_",""),
                    (row.axis_mae,row.ici_mae),
                    xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set(xlabel="10-seed mean Dev 3-axis MAE",
           ylabel="10-seed mean Dev ICI MAE",
           title="Compact-policy MAE / ICI / feature-count trade-off")
    ax.grid(alpha=.25)
    save(fig,"07_06_compact_mae_ici_pareto")
    return figures


def load_selected_seed(ctx,lock,seed):
    """Read the deployed Stage07 member, not a model with merely the same K."""
    indices=tuple(lock["selected_indices"])
    if (not indices or indices!=tuple(sorted(set(indices)))
            or indices[0]<0 or indices[-1]>=FEATURE_COUNT):
        raise RuntimeError("Invalid Stage07 selected original feature indices")
    if lock["selected_mode"]=="original":
        if indices!=tuple(range(FEATURE_COUNT)):
            raise RuntimeError("Original Stage07 model must retain all features")
        return ctx["m06"].load_policy_seed(RESULT06,ctx["locked_policy"],seed)
    if lock["selected_mode"]!="compact":
        raise RuntimeError("Stage07 selected an invalid deployment mode")
    selected=next(row for row in lock["compact_candidate_review"]["candidates"]
                  if row["policy"]==lock["selected_policy"])
    label=selected["same_subset_as"] or selected["policy"]
    subset=Subset(label,None,indices,
                  tuple(i for i in range(FEATURE_COUNT) if i not in indices),
                  pd.DataFrame(),0)
    changed=adapt_inputs(ctx["full_fold"],subset,"compact")
    folder=_candidate_folder(RESULTS,subset,"compact")
    name=f"B-MTGNN__s{seed}"
    meta=json_read(folder/(name+".json"))
    pred,ck=(folder/(name+ext) for ext in (".npz",".pt"))
    signature=run_signature(run_contract(ctx))
    if (meta["contract_sha256"]!=signature or meta["seed"]!=seed
            or meta["input_sha256"]!=dataset_sha(changed)
            or meta["prediction_sha256"]!=sha(pred)
            or meta["checkpoint_sha256"]!=sha(ck)):
        raise RuntimeError("Stage07 selected compact member failed verification")
    saved=torch.load(ck,map_location="cpu",weights_only=True)
    if (saved["contract_sha256"]!=signature or saved["seed"]!=seed
            or saved["feature_count"]!=len(indices)):
        raise RuntimeError("Stage07 selected checkpoint metadata mismatch")
    with np.load(pred,allow_pickle=False) as bank:
        prediction=bank["prediction"].copy()
    return prediction,saved["state_dict"]


def write_selected(root,ctx,subsets,masked_rows,
                   chosen:Subset,final_mode,final_metrics,
                   signature,figures,compact_examination):
    """Export only the chosen K INPUT columns, never open Stage03 Test."""
    if not chosen.kept or len(chosen.kept)+len(chosen.dropped)!=FEATURE_COUNT:
        raise RuntimeError("Invalid final greedy subset partition")
    names=ctx["panel"].features
    selected=[names[i] for i in chosen.kept]
    removed=[names[i] for i in chosen.dropped]
    if final_mode=="original":
        if chosen.label!="no_removal" or len(chosen.kept)!=FEATURE_COUNT:
            raise RuntimeError("Original setting must retain all Stage03 features")
    elif final_mode!="compact":
        raise RuntimeError("Never deploy 157-dimensional masked-input model")
    chosen.mapping.to_csv(
        root/"selected_feature_mapping.csv",
        index=False,encoding="utf-8-sig")
    pd.DataFrame([
        dict(original_index=i,feature=names[i],role=role(i))
        for i in chosen.kept]).to_csv(
            root/"selected_features.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        dict(original_index=i,feature=names[i],role=role(i),
             representative=chosen.mapping.iloc[i]["representative"])
        for i in chosen.dropped],
        columns=["original_index","feature","role","representative"]
    ).to_csv(root/"removed_features.csv",index=False,encoding="utf-8-sig")
    summary=dict(
        stage=7,status="LOCKED",
        selected_policy=chosen.label,selected_mode=final_mode,
        original_features=FEATURE_COUNT,
        selected_feature_count=len(chosen.kept),
        removed_feature_count=len(chosen.dropped),
        selected_indices=list(chosen.kept),selected_feature_names=selected,
        removed_feature_names=removed,
        baseline_157_dev=selected_baseline_scores(ctx),
        final_dev=final_metrics,
        baseline_parameter_count=ctx["baseline_parameters"],
        selected_parameter_count=int(final_metrics.get(
            "parameters",ctx["baseline_parameters"])),
        saved_parameter_count=int(
            ctx["baseline_parameters"]-final_metrics.get(
                "parameters",ctx["baseline_parameters"])),
        masked_comparisons=masked_rows,
        compact_candidate_review=compact_examination,
        stage06_selected_policy=ctx["locked_policy"].id,
        stage06_model_input_sha256=ctx["upstream06_input_sha256"],
        stage06_lock_sha256=ctx["upstream06_lock_sha256"],
        stage07_signature=signature,
        original_data_sha256=ctx["panel_hashes"],
        gnn_models=["B-MTGNN"],seeds=list(SEEDS),epochs=ctx["epochs"],
        lookback_quarters=ctx["lookback"],horizon_quarters=HORIZON,
        output=ctx["output_mode"],train="2006Q1–2019Q4",
        dev="2020Q1–2022Q4",frozen=True,
        greedy_pairwise_only=True,clustering_used=False,
        no_cross_role_feature_pruning=True,
        main_input_only=True,
        prior_context_graph_and_last_unchanged=True,
        original_42_target_history_still_accessible_via_fixed_context=True,
        all_preprocessing_statistics_fit_from_train=True,
        test_opened=False,
        dev_used_for_selection_not_independent_test=True,
        score_semantics="INDIVIDUAL_SEED_MAE_THEN_MEAN",
        compact_initialization="SUBSET_FULL_MODEL_INITIAL_WEIGHTS",
        pareto_results="compact_pareto.csv",
        block_sensitivity_results="correlations/block_sensitivity.csv",
        figures=figures,
        completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(root/"selected_features.json",summary)
    atomic_npz(
        root/"selected_feature_indices.npz",
        indices=np.asarray(chosen.kept,dtype=np.int32),
        original_count=np.asarray(FEATURE_COUNT),
        names=np.asarray(selected))
    # Do not write raw Test/full Train labels or a fake final Test artifact.
    return summary


def run(args):
    if args.restart and not args.run:
        raise ValueError("--restart requires --run")
    if args.status and not (RESULTS/"run_manifest.json").is_file():
        print("07_GREEDY_READY_NO_EXPERIMENT_YET",flush=True)
        return None
    if args.dry_run:
        print("07_GREEDY_NO_CLUSTERING")
        print("ORIGINAL 59 NODES, 157 FEATURES: 42 HIST + 24 NODE + 91 GLOBAL")
        print("CORRELATION FIT: ONLY Train 2006–2019 ORIGINAL RAW INPUT")
        print("PAIRWISE |rho| thresholds:",THRESHOLDS)
        print("BOOTSTRAP",BOOTSTRAPS,"4-QUARTER BLOCKS; lower q10 & sign >=90%")
        print("GREEDY ONLY: drop if directly redundant with ALREADY KEPT peer")
        print("B-MTGNN SELECTED MODE/L, CORE=20, SEEDS",SEEDS)
        print("75 EPOCHS; SAME FULL-TRAIN SUPPORT; 2020-22 Dev; seed-wise MAE")
        print("MASKED 157-SLOT DIAGNOSTIC + ALL FIVE TRUE COMPACT K FITS")
        print("KEEP 157 OTHERWISE; TEST NEVER OPENED")
        return None
    ctx=upstream()
    contract=run_contract(ctx)
    signature=run_signature(contract)
    if args.restart:
        ctx["m04"].reset_stage_results(RESULTS)
    path=RESULTS/"run_manifest.json"
    previous=json_read(path) if path.is_file() else None
    if previous is not None and previous.get("signature")!=signature:
        raise RuntimeError(
            "Stage07 source/upstream/data changed. Do not combine cached runs")
    if args.status:
        print(
            "07_GREEDY_STATUS",
            previous.get("status") if previous else "READY",
            "selected",previous.get("selected_policy") if previous else None,
            "masked_jobs",len(list((RESULTS/"models").glob(
                "*__masked/*.json"))),
            "compact_jobs",len(list((RESULTS/"models").glob(
                "*__compact/*.json"))),
            "test_opened",False,flush=True)
        return previous
    if not args.analyze_only and not args.run:
        raise ValueError("Supply --analyze-only or --run")
    if previous and previous.get("status")=="PASS" and not args.analyze_only:
        print("07_ALREADY_PASS",previous["selected_policy"],flush=True)
        return json_read(RESULTS/"selected_features.json")
    started=time.perf_counter()
    RESULTS.mkdir(parents=True,exist_ok=True)
    manifest=dict(
        stage=7,status="RUNNING",
        signature=signature,contract=contract,
        started_utc=(previous or {}).get(
            "started_utc",datetime.now(timezone.utc).isoformat()),
        no_clustering=True,test_opened=False,
        resume="py -u Sources/07_correlated_feature_removal.py --run")
    atomic_json(path,manifest)
    torch.set_num_threads(4)
    audit=cached_audit(ctx,RESULTS,signature)
    subsets=candidate_subsets(audit,ctx,RESULTS/"subsets")
    sensitivity_file=save_bootstrap_sensitivity(
        ctx,RESULTS,audit,subsets)
    for s in subsets:
        print("07_GREEDY_CANDIDATE",s.label,
              "kept",len(s.kept),"removed",len(s.dropped),
              "stable_edges",s.stable_edge_count,flush=True)
    if args.analyze_only:
        manifest.update(
            status="ANALYZED_TRAIN_ONLY",
            bootstrap_sensitivity=sensitivity_file,
            candidate_feature_counts={
                s.label:len(s.kept) for s in subsets},
            elapsed_seconds=time.perf_counter()-started)
        atomic_json(path,manifest)
        print("07_CORRELATION_ANALYSIS_PASS_NO_GNN_TRAIN",flush=True)
        return manifest
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device=torch.device(args.device)
    record_by_kept={}
    rows=[]
    current_baseline=selected_baseline_scores(ctx)
    for subset in subsets:
        key=subset.kept
        alias=None
        if key in record_by_kept:
            metric=record_by_kept[key]
            alias=metric["source_label"]
            result=metric["scores"]
        else:
            result=run_subset(
                ctx,RESULTS,subset,"masked",signature,device)
            record_by_kept[key]=dict(
                source_label=subset.label,scores=result)
        candidate=selection_safeguards(result,current_baseline)
        row=dict(
            policy=subset.label,
            threshold=("none" if subset.threshold is None else subset.threshold),
            mode="masked_157",
            kept_count=len(subset.kept),
            removed_count=len(subset.dropped),
            parameters=int(result["parameters"]),
            stable_edge_count=subset.stable_edge_count,
            same_subset_as=alias,
            **{k:result[k] for k in (
                "axis_mae","ici_mae","profitability_mae",
                "productivity_mae","growth_mae",
                "2020_mae","2021_mae","2022_mae",
                "axis_mae_seed_sd","ici_mae_seed_sd")},
            **candidate)
        rows.append(row)
        print("07_MASKED_RESULT",subset.label,
              "K",len(subset.kept),"MAE",round(result["axis_mae"],6),
              "ICI",round(result["ici_mae"],6),
              "YEARS",candidate["years_improved"],
              "QUALIFIED",candidate["qualified"],
              ("REUSED "+alias if alias else ""),flush=True)
        pd.DataFrame(rows).to_csv(
            RESULTS/"masked_threshold_comparison_partial.csv",
            index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(
        RESULTS/"masked_threshold_comparison.csv",
        index=False,encoding="utf-8-sig")
    compact_results=[]
    compact_by_kept={}
    for candidate in subsets[1:]:
        print("07_COMPACT_INDEPENDENT_CANDIDATE",candidate.label,
              "K",len(candidate.kept),flush=True)
        alias=compact_by_kept.get(candidate.kept)
        if alias is not None:
            compact=alias["scores"]
        elif candidate.dropped:
            compact=run_subset(
                ctx,RESULTS,candidate,"compact",signature,device)
        else:
            compact=dict(current_baseline,
                         parameters=ctx["baseline_parameters"])
        if alias is None:
            compact_by_kept[candidate.kept]=dict(
                label=candidate.label,scores=compact)
        compact_gate=selection_safeguards(compact,current_baseline)
        record=dict(
            policy=candidate.label,
            kept_count=len(candidate.kept),
            removed_count=len(candidate.dropped),
            parameters=int(compact["parameters"]),
            parameter_savings=int(
                ctx["baseline_parameters"]-compact["parameters"]),
            same_subset_as=alias["label"] if alias is not None else None,
            **{k:compact[k] for k in (
                "axis_mae","ici_mae","profitability_mae",
                "productivity_mae","growth_mae",
                "2020_mae","2021_mae","2022_mae",
                "axis_mae_seed_sd","ici_mae_seed_sd")},
            **compact_gate)
        compact_results.append(record)
        print("07_COMPACT_RESULT",candidate.label,
              "K",len(candidate.kept),
              "MAE",round(record["axis_mae"],6),
              "QUALIFIED",record["qualified"],flush=True)
        pd.DataFrame(compact_results).to_csv(
            RESULTS/"compact_comparison_partial.csv",
            index=False,encoding="utf-8-sig")
    winner,final=select_compact_candidate(
        subsets,compact_results,current_baseline)
    mode="original" if winner is subsets[0] else "compact"
    examination=dict(
        candidates=compact_results,
        compact_adopted=(mode=="compact"),
        selected_policy=winner.label,
        masked_diagnostic_used_for_compact_selection=False,
        trained_compact_models=len(SEEDS)*sum(
            bool(x["scores"].get("removed_count",0))
            for x in compact_by_kept.values()))
    pd.DataFrame(compact_results).to_csv(
        RESULTS/"compact_confirmation.csv",
        index=False,encoding="utf-8-sig")
    pareto=pd.DataFrame([
        dict(policy="no_removal",kept_count=FEATURE_COUNT,
             parameters=ctx["baseline_parameters"],
             axis_mae=current_baseline["axis_mae"],
             ici_mae=current_baseline["ici_mae"]),
        *[dict(policy=r["policy"],kept_count=r["kept_count"],
               parameters=r["parameters"],axis_mae=r["axis_mae"],
               ici_mae=r["ici_mae"]) for r in compact_results]])
    pareto["nondominated"]=[
        not any(
            q.kept_count<=p.kept_count and
            q.axis_mae<=p.axis_mae and q.ici_mae<=p.ici_mae and
            (q.kept_count<p.kept_count or q.axis_mae<p.axis_mae or
             q.ici_mae<p.ici_mae)
            for q in pareto.itertuples(index=False))
        for p in pareto.itertuples(index=False)]
    pareto.to_csv(
        RESULTS/"compact_pareto.csv",index=False,encoding="utf-8-sig")
    figs=plots(RESULTS,ctx,audit,subsets,rows,
              compact_results,winner,mode,final)
    selected=write_selected(
        RESULTS,ctx,subsets,rows,winner,mode,final,
        signature,figs,examination)
    manifest.update(
        status="PASS",selected_policy=winner.label,
        selected_mode=mode,kept=len(winner.kept),
        removed=len(winner.dropped),
        masked_candidates=len(rows),
        compact_candidates=len(compact_results),
        distinct_masked_feature_sets=len(record_by_kept),
        distinct_compact_feature_sets=len(compact_by_kept),
        bootstrap_sensitivity=sensitivity_file,
        baseline_parameters=ctx["baseline_parameters"],
        selected_parameters=int(final.get(
            "parameters",ctx["baseline_parameters"])),
        compact_confirmation=examination,
        figures=figs,
        elapsed_seconds=time.perf_counter()-started,
        completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(path,manifest)
    print("07_GREEDY_COMPLETE",winner.label,
          "mode",mode,"kept",len(winner.kept),
          "removed",len(winner.dropped),
          "MAE",round(final["axis_mae"],6),
          "06_SELECTED_MAE",round(current_baseline["axis_mae"],6),
          "PLOTS",len(figs),flush=True)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--analyze-only",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--restart",action="store_true")
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--device",choices=("cpu","cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    run(args)


if __name__=="__main__":
    main()
