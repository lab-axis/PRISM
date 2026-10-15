#!/usr/bin/env python3
"""PRISM 04 — B-MTGNN robustness and target-forecasting formulations.

Train 2006–2019, Dev 2020–2022. B-MTGNN ONLY, ten matched random seeds,
and three target formulations: Direct, Residual, Structural.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"Results/04_bmtgnn_benchmark"
DEFAULT_SEEDS=tuple(range(2026,2036))
GNN_MODELS=("B-MTGNN",)
MODELS=GNN_MODELS
EXTRA_NAIVE_BASELINES=("HistoricalMean","LinearDrift")
NAIVE_MODELS=("Persistence","SeasonalNaive","HistoricalMean","LinearDrift")
STAGE04_OUTPUT_MODES=("direct","residual","structural")
STAGE04_ACTIVE_EPOCHS=75
STAGE04_ROBUST_MAX_EPOCHS=100
STAGE04_ROBUST_SNAPSHOTS=(50,75,100)
STAGE04_BMTGNN_MC_SAMPLES=30
STAGE04_ACTIVE_WIDTH=16
STAGE04_FIXED_ALPHA=(.25,.1,.25)
STAGE04_ACTIVE_ALPHA_GRID=(0.,.1,.2,.25,.3,.5,.75,1.)
STAGE04_ALPHA_POLICY="FROZEN_FULL_TRAIN_YEAR_PAIRED_ONE_SE"
STAGE04_ALPHA_CALIBRATION_END=56
FIXED_LEARNING_RATE=.001
TARGETS=["target_profitability","target_productivity","target_growth"]
IDS=["quarter","node_key","node_name"]
LOOKBACK,HORIZON=12,1
NODE_COUNT=59
TRAIN_COUNT=56
DEV_COUNT=12
TOTAL_COUNT=TRAIN_COUNT+DEV_COUNT
STAGE04_COMPONENT_SOURCE=ROOT/"Data/01_target_data_collection/latest_sources.csv.gz"
STAGE04_TARGET_BOUNDS=ROOT/"Results/03_integrated_dataset/target_score_bounds.csv"
LEARNING_RECIPE="TRAIN_MEDIAN_ZERO_RESIDUAL_JOINT_MAE"

def fixed_preprocessing_policy(model):
    if model!="B-MTGNN":
        raise ValueError("04 trains only B-MTGNN")
    return "standard","mask"

@dataclass
class Panel:
    x: np.ndarray
    y: np.ndarray
    quarters: list[str]
    nodes: list[str]
    names: list[str]
    features: list[str]
    train_count: int
    hashes: dict[str, str]

def load_panel(root: Path) -> Panel:
    """Load strictly Train and Dev, never any later split."""
    tables, hashes = [], {}
    partitions=[
        ("train",("2006Q1","2019Q4",TRAIN_COUNT*NODE_COUNT)),
        ("val",("2020Q1","2022Q4",DEV_COUNT*NODE_COUNT))]
    for name, expected in partitions:
        path = root / "Data/03_integrated_dataset" / f"raw_{name}.csv"
        hashes[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, dtype={"quarter": str, "node_key": str}, low_memory=False)
        if (frame.quarter.min(), frame.quarter.max(), len(frame)) != expected:
            raise ValueError(f"Unexpected Stage 03 {name} partition")
        tables.append(frame)
    if any(tables[0].columns.tolist()!=part.columns.tolist()
           for part in tables[1:]):
        raise ValueError("Train/Dev schema mismatch")
    frame = pd.concat(tables, ignore_index=True)
    features = [c for c in frame if c not in IDS + TARGETS + ["target_ici"]]
    if features[:3] != ["history_profitability", "history_productivity", "history_growth"]:
        raise ValueError("Three previous-quarter target features must be first")
    nodes = sorted(frame.node_key.unique())
    quarters = pd.period_range(
        "2006Q1","2022Q4",freq="Q").astype(str).tolist()
    length=len(quarters)
    engineered_count = len(features) - 115
    if (len(nodes) != NODE_COUNT or engineered_count != 42
            or any(not c.startswith(("history_", "targethist_")) for c in features[:engineered_count])
            or any(c.startswith(("history_", "targethist_")) for c in features[engineered_count:])
            or frame.duplicated(["quarter", "node_key"]).any()):
        raise ValueError("Require 59 industries, 42 Target-history and 115 original predictors")
    index = pd.MultiIndex.from_product([quarters, nodes], names=["quarter", "node_key"])
    frame = frame.set_index(["quarter", "node_key"]).reindex(index)
    y = frame[TARGETS].to_numpy(float).reshape(length, NODE_COUNT, 3)
    x = frame[features].to_numpy(float).reshape(length, NODE_COUNT, len(features))
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError("Nonfinite targets or infinite predictors")
    if not np.allclose(frame.target_ici.to_numpy(), y.mean(-1).reshape(-1)):
        raise ValueError("Stage 03 ICI identity mismatch")
    names = frame["node_name"].groupby(level="node_key").first().reindex(nodes).tolist()
    return Panel(x, y, quarters, nodes, names, features, TRAIN_COUNT, hashes)

def fit_normalizer(x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit only observed Train entries, deterministic across memory layouts."""
    if kind!="standard":
        raise ValueError(kind)
    # 2/3D ndarray views vs copies may sum in a different floating-point
    # order. Flatten deterministically first, so poisoning unused Dev rows
    # cannot appear to perturb Train normalization at the 1e-16 scale.
    values=np.ascontiguousarray(
        np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center, scale = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-8), scale, 1.)
    return center, scale

def past_scores(panel: Panel, origins: np.ndarray) -> np.ndarray:
    """For prediction of Y[t], Stage03 row t's first 3 fields use t-1 only.

    Never supply the other 154 fields of row t to a forecast of Y[t].
    """
    origins = np.asarray(origins, dtype=int)
    if origins.ndim != 1 or len(origins) == 0 or origins.min() < 1 or origins.max() >= len(panel.quarters):
        raise ValueError("Invalid one-quarter-ahead prediction origins")
    scores = panel.x[origins, :, :3]
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete lagged Target input at prediction origin")
    return scores.copy()

def origin_lag_scores(panel: Panel, origins: np.ndarray, distance: int) -> np.ndarray:
    """Stage03's profitability history is re-evaluated at each forecast origin.

    For a next-quarter prediction at t, lag d always comes from the
    strictly-past bank stored in row t, not from row t-d+1 where the
    completed-year GOS benchmark may not yet have been incorporated.
    """
    if distance == 1:
        return past_scores(panel, origins)
    if distance not in (2, 4, 5):
        raise ValueError("Only registered as-of history depths are supported")
    origins = np.asarray(origins, dtype=int)
    if distance in (2, 4):
        ids = [panel.features.index(f"targethist_{axis}_lag{distance:02d}")
               for axis in ("profitability", "productivity", "growth")]
        values = panel.x[origins][:, :, ids]
    else:
        change_ids = [panel.features.index(f"targethist_{axis}_change04")
                      for axis in ("profitability", "productivity", "growth")]
        values = past_scores(panel, origins) - panel.x[origins][:, :, change_ids]
    if values.ndim != 3 or values.shape[1:] != (NODE_COUNT, 3) or not np.isfinite(values).all():
        raise ValueError("Missing registered as-of historical score")
    return values.copy()

def known_score_context(panel: Panel, origins: np.ndarray,
                        y_center: np.ndarray, y_scale: np.ndarray) -> np.ndarray:
    """Normalized [last score, last-lag2, last-lag5] from the SAME row t.

    All three axes and all B-MTGNN target formulations receive the same
    nine observed historical values. No current-quarter Y[t] is included.
    """
    last = origin_lag_scores(panel, origins, 1)
    lag2 = origin_lag_scores(panel, origins, 2)
    lag5 = origin_lag_scores(panel, origins, 5)
    result = np.concatenate(
        [(last-y_center)/y_scale, (last-lag2)/y_scale,
         (last-lag5)/y_scale], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite strictly historical next-quarter context")
    return result

def row_normalize(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1,keepdim=True).clamp_min(1e-8)

class MixHop(nn.Module):
    def __init__(self, width, depth=2, alpha=.05):
        super().__init__()
        self.depth, self.alpha = depth, alpha
        self.out = nn.Conv2d((depth + 1) * width, width, 1)

    def forward(self, x, a):
        a = row_normalize(a + torch.eye(a.size(0), device=a.device))
        h, parts = x, [x]
        for _ in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("bcnt,mn->bcmt", h, a)
            parts.append(h)
        return self.out(torch.cat(parts, 1))

class Inception(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.parts = nn.ModuleList([nn.Conv2d(width, width // 4, (1, k), dilation=(1, dilation))
                                    for k in (2, 3, 6, 7)])

    def forward(self, x):
        parts = [layer(x) for layer in self.parts]
        return torch.cat([z[..., -parts[-1].size(-1):] for z in parts], 1)

class MTGNNCore(nn.Module):
    """MTGNN-style learned graph and gated temporal convolutional backbone."""
    def __init__(self, nodes, width, lookback, horizon):
        super().__init__()
        self.nodes, self.horizon, self.rf = nodes, horizon, 19
        self.e1, self.e2 = (nn.Parameter(torch.randn(nodes, 8) * .1) for _ in range(2))
        self.l1, self.l2 = nn.Linear(8, 8), nn.Linear(8, 8)
        self.start = nn.Conv2d(width, width, 1)
        self.filters = nn.ModuleList([Inception(width, d) for d in (1, 2)])
        self.gates = nn.ModuleList([Inception(width, d) for d in (1, 2)])
        self.fwd = nn.ModuleList([MixHop(width) for _ in range(2)])
        self.rev = nn.ModuleList([MixHop(width) for _ in range(2)])
        length = max(lookback, self.rf)
        widths = [length - 6, length - 18]
        self.skip = nn.ModuleList([nn.Conv2d(width, 2 * width, (1, w)) for w in widths])
        self.norm = nn.ModuleList([nn.LayerNorm([width, nodes, w]) for w in widths])
        self.skip0 = nn.Conv2d(width, 2 * width, (1, length))
        self.skipend = nn.Conv2d(width, 2 * width, (1, widths[-1]))
        self.out = nn.Sequential(nn.ReLU(), nn.Conv2d(2 * width, 4 * width, 1), nn.ReLU(),
                                 nn.Conv2d(4 * width, horizon * 3, 1))
        self.last_graph = None

    def graph(self):
        e1, e2 = torch.tanh(3 * self.l1(self.e1)), torch.tanh(3 * self.l2(self.e2))
        a = F.relu(torch.tanh(3 * (e1 @ e2.T - e2 @ e1.T)))
        idx = a.topk(min(8, self.nodes), dim=1).indices
        return a * torch.zeros_like(a).scatter_(1, idx, 1.)

    def forward(self, x, unused):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, (max(0, self.rf - x.size(-1)), 0))
        a = self.graph()
        self.last_graph = a.detach()
        skip = self.skip0(F.dropout(x, .1, self.training))
        x = self.start(x)
        for filt, gate, fwd, rev, sk, norm in zip(
                self.filters, self.gates, self.fwd, self.rev, self.skip, self.norm):
            residual = x
            x = torch.tanh(filt(x)) * torch.sigmoid(gate(x))
            x = F.dropout(x, .1, self.training)
            skip = skip + sk(x)
            x = norm(fwd(x, a) + rev(x, a.T) + residual[..., -x.size(-1):])
        z = self.out(skip + self.skipend(x)).squeeze(-1)
        return z.reshape(x.size(0), self.horizon, 3, self.nodes).permute(0, 1, 3, 2)

class ForecastModel(nn.Module):
    """Common information set, raw-zero control, optional separate mask adapter.

    Every neural model receives the 42 Target-derived features and the original
    115 external predictors, in the exact order supplied by Stage 03.
    Missingness is NOT written as new source features. For mask runs, a separate
    learned projection distinguishes imputed raw zeros from observed real zeros.
    The same projection parameters exist in zero-control runs but receive zeros.
    """
    def __init__(self, name, features, nodes, width, lookback, horizon):
        super().__init__()
        if name!="B-MTGNN":
            raise ValueError("Only B-MTGNN is trained by Stage04")
        self.name = name
        self.values = nn.Linear(features, width)
        self.missing = nn.Linear(features, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.core = MTGNNCore(nodes,width,lookback,horizon)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def economic_components(panel: Panel) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct next-quarter baselines from Stage01's historical components.

    Profitability MUST use per-origin Stage03 history. Once a past year ends
    its annual GOS becomes usable, so t-1 and t-2 must be reconstructed
    using the same forecast origin. Never use retrospective Target labels.
    The latest revised official source values remain the fixed data policy.
    Only the already completed 2005 annual GOS is read to build the
    2006Q1-2008Q4 calibration history.
    """
    end=panel.quarters[-1]
    periods = pd.period_range("2005Q1",end,freq="Q").astype(str)
    months = pd.period_range(
        "2005-01",pd.Period(end,freq="Q").end_time.strftime("%Y-%m"),
        freq="M").astype(str)
    source = []
    usecols = ["provider", "series", "frequency", "period",
               "node_key", "units", "value"]
    for chunk in pd.read_csv(
            STAGE04_COMPONENT_SOURCE, usecols=usecols, chunksize=16384):
        selected = chunk.loc[
            (chunk.provider.eq("BEA") &
             ((chunk.frequency.eq("quarterly") &
               chunk.series.isin(("current_go", "current_va", "real_va", "real_go")) &
               chunk.period.isin(periods)) |
              (chunk.frequency.eq("annual") & chunk.series.eq("gos") &
               chunk.period.eq("2005")))) |
            (chunk.series.isin(("employment","employment_523_525_aggregate")) &
             chunk.frequency.eq("monthly") & chunk.period.isin(months))]
        if not selected.empty:
            source.append(selected)
    if not source:
        raise RuntimeError("Required historical economic components missing")
    source = pd.concat(source, ignore_index=True)

    def bea(name, frequency, labels):
        indexed = (source.loc[source.provider.eq("BEA") &
                              source.series.eq(name) &
                              source.frequency.eq(frequency)]
                   .groupby(["period", "node_key"]).value.sum().unstack()
                   .reindex(index=labels, columns=panel.nodes))
        if indexed.isna().any().any():
            raise RuntimeError(f"Incomplete historical BEA {name}")
        return indexed.to_numpy(dtype=np.float64)

    nominal_go = bea("current_go", "quarterly", periods)
    current_va = bea("current_va", "quarterly", periods[:4])
    annual_gos_2005 = bea("gos", "annual", ["2005"])[0]
    real_va = bea("real_va", "quarterly", periods)
    real_go = bea("real_go", "quarterly", periods)

    employed = source.loc[source.series.eq("employment")].copy()
    employed["thousands"] = np.where(
        employed.units.eq("persons"),employed.value/1000.,employed.value)
    monthly = (employed.groupby(["period","node_key"]).thousands.sum()
               .unstack().reindex(index=months,columns=panel.nodes))
    if monthly.isna().any().any() or (monthly<=0).any().any():
        raise RuntimeError("Incomplete Stage01 historical employment")
    employment = monthly.to_numpy(dtype=np.float64).reshape(
        len(periods),3,NODE_COUNT).mean(1)
    combined = (source.loc[source.series.eq("employment_523_525_aggregate")]
                .set_index("period").value.reindex(months))
    if combined.isna().any() or (combined<=0).any():
        raise RuntimeError("Incomplete Stage01 funds employment aggregate")
    funds, securities = (panel.nodes.index("funds_trusts"),
                         panel.nodes.index("securities"))
    employment[:,funds] = combined.to_numpy(dtype=np.float64).reshape(
        len(periods),3).mean(1)
    real_va = real_va.copy()
    real_va[:, funds] += real_va[:, securities]
    denominators = np.stack(
        [nominal_go[4:], employment[4:], real_go[:-4]],
        axis=-1)
    bounds = pd.read_csv(STAGE04_TARGET_BOUNDS)
    ordered = ("profitability","productivity","growth")
    if not bounds.reference_end.eq("2019Q4").all():
        raise RuntimeError("Score bounds are not Train-only")
    lo, hi = [], []
    for axis in ordered:
        part = bounds.loc[bounds.component.eq(axis)]
        if part.empty or part["min"].nunique()!=1 or part["max"].nunique()!=1:
            raise RuntimeError(f"Invalid frozen score bounds for {axis}")
        lo.append(float(part["min"].iloc[0]))
        hi.append(float(part["max"].iloc[0]))
    lo, hi = np.array(lo), np.array(hi)
    # All four quarters of 2005 and their annual GOS are past information
    # for EVERY 2006+ forecast. Needed to use the complete 2006-2019 Train
    # for alpha calibration, without imposing the neural 12-quarter window.
    base_va = current_va.mean(axis=0)
    if not np.isfinite(base_va).all() or (np.abs(base_va) < 1e-12).any():
        raise RuntimeError("Invalid 2005 VA denominator for alpha calibration")
    gos_2005 = current_va * (annual_gos_2005 / base_va)[None, :]
    pretrain_num = np.stack(
        [gos_2005, real_va[:4], real_go[:4]], axis=-1)
    # 2005 growth denominators need 2004 quarterlies, which are unavailable,
    # but are NEVER used: a forecast of 2006+ growth substitutes the known
    # prior-year same-quarter GO from den[t,:,2] instead.
    pretrain_den = np.stack(
        [nominal_go[:4], employment[:4], np.ones_like(real_go[:4])], axis=-1)
    if (not np.isfinite(pretrain_num).all()
            or not np.isfinite(pretrain_den).all()
            or not (pretrain_den > 0).all()):
        raise RuntimeError("Invalid 2005 historical component prefix")
    # Stage03 x[t+1,0] is the already-computed profitability for quarter t,
    # as known to the NEXT forecast origin. No duplicate GOS estimator here.
    # The final row is only a finite placeholder, never used as a past
    # component for any 2006-2022 prediction origin.
    historic_ratio = (lo[0] +
                      (panel.x[1:, :, 0] / 100.) * (hi[0] - lo[0]))
    historical_gos = np.empty_like(nominal_go[4:])
    historical_gos[:-1] = historic_ratio * denominators[:-1, :, 0]
    historical_gos[-1] = historical_gos[-2]
    numerators = np.stack(
        [historical_gos, real_va[4:], real_go[4:]], axis=-1)
    if (not np.isfinite(numerators).all()
            or not np.isfinite(denominators).all()
            or not (denominators > 0).all()):
        raise RuntimeError("Invalid economic component decomposition")
    raw_rebuilt = numerators/denominators
    raw_rebuilt[..., 2] -= 1.
    scores = 100. * (raw_rebuilt-lo)/(hi-lo)
    if not np.allclose(scores[:-1,:,0],panel.x[1:,:,0],
                       atol=1e-9,rtol=1e-10):
        raise RuntimeError("Structural profitability disagrees with Stage03 causal feature")
    if not np.allclose(scores[...,1:],panel.y[...,1:],atol=1e-9,rtol=1e-10):
        raise RuntimeError("Structural productivity/growth identities changed")
    info = dict(
        target_definitions=[
            "Origin-specific Stage03 completed-year/provisional GOS / nominal gross output",
            "real value added / matched quarterly employment",
            "real gross output / same-quarter prior-year gross output - 1"],
        fixed_train_min=lo.tolist(), fixed_train_max=hi.tolist(),
        pretrain_component_num=pretrain_num,
        pretrain_component_den=pretrain_den,
        # Current origin's correct t-1 and t-2 snapshots. Old static
        # num[t-2] can be stale when a year-end annual GOS becomes available.
        profitability_asof_lag1=panel.x[:, :, 0].copy(),
        profitability_asof_lag2=panel.x[
            :, :, panel.features.index("targethist_profitability_lag02")].copy(),
        past_profitability_input_max_abs=float(
            np.abs(scores[:-1,:,0]-panel.x[1:,:,0]).max()),
        unchanged_productivity_growth_max_abs=float(
            np.abs(scores[...,1:]-panel.y[...,1:]).max()),
        input_source_period=["2005Q1",end],
        official_data_policy="Use latest revised official values as the "
                             "authoritative retrospective reference.")
    return numerators, denominators, info

def component_structural_prior(
        num: np.ndarray, den: np.ndarray, source_info: dict,
        origins: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Apply identical component extrapolation family to the three axes.

    Extrapolate unknown N[t] from N[t-1],N[t-2]. For unknown denominators
    use exponential extrapolation of positive D[t-1]/D[t-2]. Growth's
    denominator D[t]=R[t-4] is already *known* and is used directly.
    Stage01's 2005 prefix supplies the two preceding native quarters for
    the earliest 2006 forecasts. It is not a 12-quarter neural input.
    Alpha is fixed using only the 2006-2019 Train labels.
    """
    indices = np.asarray(origins, dtype=int)
    weights = np.asarray(alpha, dtype=np.float64)
    if (indices.ndim != 1 or indices.size == 0
            or indices.min() < 0 or indices.max() >= len(num)
            or weights.shape != (3,) or not np.isfinite(weights).all()):
        raise ValueError("Invalid structural forecast origins or gain")
    if not set(weights.tolist()).issubset(set(STAGE04_ACTIVE_ALPHA_GRID)):
        raise ValueError("Structural component extrapolation is not Train-locked")
    prefix_num = np.asarray(source_info["pretrain_component_num"])
    prefix_den = np.asarray(source_info["pretrain_component_den"])
    if (prefix_num.shape != (4, NODE_COUNT, 3)
            or prefix_den.shape != prefix_num.shape):
        raise ValueError("Expected exactly four 2005 prehistory quarters")
    historical_num = np.concatenate((prefix_num, num), axis=0)
    historical_den = np.concatenate((prefix_den, den), axis=0)
    offsets = indices + 4
    N = historical_num[offsets-1].copy()
    diff = N-historical_num[offsets-2]
    # Reconstruct both historic GOS numerators from Stage03's two as-of
    # scores at THIS SAME forecast origin. Completed annual GOS can improve
    # even the t-2 estimate at the start of a new year. Do NOT use num[t-2,0]
    # from an older snapshot, and never use the ex-post Target y[t-1].
    low = np.asarray(source_info["fixed_train_min"])
    high = np.asarray(source_info["fixed_train_max"])
    lag1 = np.asarray(source_info["profitability_asof_lag1"])[indices]
    lag2 = np.asarray(source_info["profitability_asof_lag2"])[indices]
    if (lag1.shape != (len(indices), NODE_COUNT)
            or lag2.shape != lag1.shape
            or not np.isfinite(lag1).all() or not np.isfinite(lag2).all()):
        raise RuntimeError("Missing origin-specific Stage03 profitability inputs")
    n1 = (low[0]+lag1*(high[0]-low[0])/100.)*historical_den[offsets-1,:,0]
    n2 = (low[0]+lag2*(high[0]-low[0])/100.)*historical_den[offsets-2,:,0]
    N[:,:,0] = n1
    diff[:,:,0] = n1-n2
    # Stable numerator extrapolation (GOS and real VA may be negative).
    # Limit the step symmetrically without using current-quarter labels.
    # Each forecast origin uses its own prior-quarter history. Do not use
    # a 2019-wide clipping statistic for earlier Train-internal origins.
    past_scale = np.stack([
        np.nanmedian(np.abs(historical_num[:origin]), axis=(0,1))
        for origin in offsets])
    historical_bound = 2. * np.maximum(np.abs(N),past_scale[:,None,:])
    N += weights[None, None, :] * np.clip(
        diff, -historical_bound, historical_bound)
    D = historical_den[offsets-1].copy()
    log_drift = np.log(historical_den[offsets-1]/historical_den[offsets-2])
    D *= np.exp(weights[None, None, :] * np.clip(log_drift, -.75, .75))
    D[..., 2] = den[indices, :, 2]  # known prior-year same-quarter GO
    ratio = N/D
    ratio[..., 2] -= 1.
    out = (100. * (ratio-low)/(high-low)).astype(np.float64)
    if out.shape != (len(indices), NODE_COUNT, 3) or not np.isfinite(out).all():
        raise RuntimeError("Malformed structure-based next-quarter prediction")
    return out

def select_train_structural_priors(panel: Panel, n, d, info,
                                   fit_end: int | None = None):
    """Freeze alpha from ALL 2006-2019 Train years, never from Dev.

    All 56 Train quarters participate in the 14 year-block paired 1-SE
    comparison, independently of the neural model's twelve-quarter lookback.
    """
    end = panel.train_count
    if (end != TRAIN_COUNT or STAGE04_ALPHA_CALIBRATION_END != TRAIN_COUNT
            or (fit_end is not None and int(fit_end) != TRAIN_COUNT)):
        raise ValueError("Stage04 alpha selection requires ALL 2006-2019 Train quarters")
    origins = np.arange(0, end)
    reference_origins = origins
    reference_years = np.array(
        [int(panel.quarters[t][:4]) for t in reference_origins])
    years = np.unique(reference_years)
    if (len(years) != 14 or years[0] != 2006 or years[-1] != 2019
            or any((reference_years == year).sum() != 4 for year in years)):
        raise ValueError("Robust 04 alpha audit requires fourteen complete Train years")
    audit = []
    year_losses = []
    for alpha in STAGE04_ACTIVE_ALPHA_GRID:
        prior = component_structural_prior(
            n, d, info, origins, np.repeat(alpha, 3))
        axis_mae = np.abs(prior-panel.y[origins]).mean((0, 1))
        reference_loss = np.abs(
            prior[:len(reference_origins)] - panel.y[reference_origins]).mean(axis=1)
        annual_loss = np.stack([
            reference_loss[reference_years == year].mean(axis=0)
            for year in years])
        year_losses.append(annual_loss)
        audit.append(dict(
            coefficient=float(alpha),
            profitability_mae=float(axis_mae[0]),
            productivity_mae=float(axis_mae[1]),
            growth_mae=float(axis_mae[2]),
            audit_train_end=panel.quarters[end-1],
            alpha_decision_period="2006-2019 (ALL Train years, frozen)",
            alpha_decision_quarters=len(reference_origins),
            alpha_decision_years=len(years),
            selection_policy=STAGE04_ALPHA_POLICY))
    table = pd.DataFrame(audit)
    yearly = np.stack(year_losses)  # candidate x independent Train-year x axis
    for axis_idx, axis in enumerate(("profitability","productivity","growth")):
        year_error = yearly[:, :, axis_idx]
        mean = year_error.mean(axis=1)
        best = int(np.argmin(mean))
        paired = year_error - year_error[best]
        se = paired.std(axis=1, ddof=1)/np.sqrt(len(years))
        eligible = mean - mean[best] <= se + 1e-12
        frozen = float(STAGE04_FIXED_ALPHA[axis_idx])
        table[f"{axis}_full_train_year_mae"] = mean
        table[f"{axis}_paired_year_se_from_best"] = se
        table[f"{axis}_within_paired_one_se"] = eligible
        table[f"{axis}_is_frozen_alpha"] = table.coefficient.eq(frozen)
        table[f"{axis}_frozen_alpha"] = frozen
    chosen = np.asarray(STAGE04_FIXED_ALPHA, dtype=float)
    if not set(chosen.tolist()).issubset(set(STAGE04_ACTIVE_ALPHA_GRID)):
        raise RuntimeError("Frozen 04 alpha must be in the audited policy grid")
    for i, axis in enumerate(("profitability", "productivity", "growth")):
        eligible = table.loc[table[f"{axis}_within_paired_one_se"], "coefficient"]
        if eligible.empty or not np.isclose(chosen[i], float(eligible.min())):
            raise RuntimeError(
                f"Frozen {axis} alpha disagrees with ALL-Train paired 1-SE rule")
    return chosen, table

class ThreeAxisStructureGNN(nn.Module):
    """Same core/features/head/loss; compare three different forecast anchors.

    Direct:     forecast normalized Y directly (zero normalized anchor).
    Residual:   forecast = Stage03 past-only score at t-1 + correction.
    Structural: forecast = 3-axis economic prior + learned correction.
    For a fixed seed and GNN, all modules have the EXACT same architecture,
    initial parameter seeds, contextual predictors and training loss.
    """
    def __init__(self, architecture, x_center, x_scale, y_center, y_scale,
                 num_features=157, nodes=NODE_COUNT, prediction_mode="structural"):
        super().__init__()
        if prediction_mode not in STAGE04_OUTPUT_MODES:
            raise ValueError(f"Unknown output mode: {prediction_mode}")
        self.architecture = architecture
        self.prediction_mode = prediction_mode
        self.base = ForecastModel(
            architecture, num_features, nodes, STAGE04_ACTIVE_WIDTH,
            LOOKBACK, HORIZON)
        self.context_adapter = nn.Sequential(
            nn.Linear(9, STAGE04_ACTIVE_WIDTH), nn.SiLU(),
            nn.Linear(STAGE04_ACTIVE_WIDTH, STAGE04_ACTIVE_WIDTH))
        self.change_head = nn.Sequential(
            nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.change_head[-1].weight)
        nn.init.zeros_(self.change_head[-1].bias)
        # Estimated from Train only; persisted with the checkpoint, not a
        # learned neural parameter and not a correction fitted on Dev.
        self.register_buffer("anchor_bias",torch.zeros(nodes,3))
        self.register_buffer("x_c", torch.as_tensor(
            x_center[:3], dtype=torch.float32))
        self.register_buffer("x_s", torch.as_tensor(
            x_scale[:3], dtype=torch.float32))
        self.register_buffer("y_c", torch.as_tensor(
            y_center, dtype=torch.float32))
        self.register_buffer("y_s", torch.as_tensor(
            y_scale, dtype=torch.float32))

    def forward(self, x, mask, graph, last, prior, context):
        if context.shape != (*last.shape[:2], 9):
            raise ValueError("Missing 9D strictly as-of Stage03 context")
        z = self.base.norm(
            self.base.values(x)+self.base.missing(mask)+
            self.context_adapter(context)[:,None])
        correction = self.base.core(z,graph)
        # The labels are ALWAYS the exact same Stage03 Y[t]. Only the
        # network-output reference differs. Direct is NOT a delta forecast.
        if self.prediction_mode == "direct":
            anchor = torch.zeros_like(last)
        elif self.prediction_mode == "residual":
            anchor = last
        else:
            anchor = prior
        return (anchor[:,None]+self.anchor_bias[None,None]
                +correction+self.change_head(context)[:,None])


def admissible_train_origins(lookback: int) -> np.ndarray:
    """Use every Train target with both L raw inputs and lag-5 context."""
    if not 1<=lookback<=20:
        raise ValueError("Lookback must be in 1..20")
    return np.arange(max(lookback,5),TRAIN_COUNT,dtype=int)


def train_anchor_biases(panel,structure,origins,center,scale):
    """Empirical median residual per industry/axis, in model coordinates.

    Compute in original float64 score units before converting to float32.
    All three formulations receive the same calibration rule; their
    anchors, not their optimization rules, remain the comparison factor.
    """
    origins=np.asarray(origins,dtype=int)
    if (origins.ndim!=1 or not len(origins) or origins.min()<5
            or origins.max()>=TRAIN_COUNT
            or not np.array_equal(origins,np.unique(origins))):
        raise ValueError("Calibration requires unique admissible Train origins")
    targets=panel.y[origins]
    anchors=(np.broadcast_to(center,targets.shape),
             past_scores(panel,origins),structure[origins])
    return np.stack([
        np.median(targets-anchor,axis=0)/scale for anchor in anchors
    ]).astype(np.float32)


def initialize_calibrated_model(model,data):
    """Call after the final core replacement, before the first optimizer step.

    Zero only the output layers: all interior parameters retain their
    original seeded initialization. Native Stage04 and padded Stage05
    cores keep their existing, different temporal architectures.
    """
    core=model.base.core
    core=core.backbone if hasattr(core,"backbone") else core
    nn.init.zeros_(core.out[-1].weight)
    nn.init.zeros_(core.out[-1].bias)
    nn.init.zeros_(model.change_head[-1].weight)
    nn.init.zeros_(model.change_head[-1].bias)
    bias=data["anchor_biases"][STAGE04_OUTPUT_MODES.index(model.prediction_mode)]
    with torch.no_grad():
        model.anchor_bias.copy_(torch.as_tensor(bias,device=model.anchor_bias.device))

def prepare_three_axis_fold(panel: Panel, structure: np.ndarray,
                            fit_end: int, val_first: int,
                            val_end: int, scaler: str = "standard",
                            missing_mode: str = "mask") -> dict:
    if (structure.shape != panel.y.shape
            or not 20 <= fit_end <= TRAIN_COUNT
            or not fit_end <= val_first < val_end <= len(panel.quarters)):
        raise ValueError("Invalid Train-only three-axis structure fold")
    if (scaler,missing_mode)!=("standard","mask"):
        raise ValueError("Only Train-fitted standard/mask B-MTGNN is supported")
    center,width=fit_normalizer(panel.x[:fit_end],"standard")
    observed = np.isfinite(panel.x)
    # Impute original NaNs with raw 0 before Train-only normalization;
    # supply a separate missingness mask.
    x = ((np.where(observed,panel.x,0.)-center)/width).astype(np.float32)
    nan_mask=(~observed).astype(np.float32)
    masks=nan_mask
    yc = panel.y[:fit_end].mean((0,1))
    ys = np.where(panel.y[:fit_end].std((0,1)) > 1e-8,
                  panel.y[:fit_end].std((0,1)),1.)
    y = ((panel.y-yc)/ys).astype(np.float32)
    structural = ((structure-yc)/ys).astype(np.float32)
    train = admissible_train_origins(LOOKBACK)
    if fit_end!=TRAIN_COUNT:
        raise ValueError("Train-internal splits are not part of this pipeline")
    val = np.arange(val_first,val_end)
    def tensors(origins,prefix):
        return {
            f"{prefix}_x":np.stack([x[o-LOOKBACK:o] for o in origins]),
            f"{prefix}_mask":np.stack([masks[o-LOOKBACK:o] for o in origins]),
            f"{prefix}_raw_nan_mask":np.stack([
                nan_mask[o-LOOKBACK:o] for o in origins]),
            f"{prefix}_y":np.stack([y[o:o+1] for o in origins]),
            f"{prefix}_last":((past_scores(panel,origins)-yc)/ys).astype(np.float32),
            f"{prefix}_context":known_score_context(panel,origins,yc,ys),
            f"{prefix}_structure":structural[origins],
        }
    output = {
        **tensors(train,"train"),**tensors(val,"val"),
        "x_center":center.astype(np.float32),
        "x_scale":width.astype(np.float32),
        "y_center":yc.astype(np.float32),
        "y_scale":ys.astype(np.float32),
        "train_origins":train,"val_origins":val,
        "anchor_biases":train_anchor_biases(panel,structure,train,yc,ys),
        "scaler_policy":scaler,"missing_policy":missing_mode,
    }
    if (output["train_y"].shape != (fit_end-LOOKBACK,1,NODE_COUNT,3)
            or output["val_y"].shape != (val_end-val_first,1,NODE_COUNT,3)
            or not np.isfinite(output["val_structure"]).all()
            or not np.allclose(
                output["val_last"][0]*ys+yc,past_scores(panel,np.array([val_first]))[0],
                atol=3e-5,rtol=3e-5)):
        raise RuntimeError("Three-axis structural Train/Val timing differs")
    return output

def joint_axis_ici_loss(pred, truth, ys):
    """Equal-weight original-unit axis MAE and ICI MAE; ICI is derived."""
    error = (pred-truth)*ys
    return (error.abs().mean()+error.mean(-1).abs().mean())/2.

@contextmanager
def isolated_snapshot_inference(model):
    """Restore every training RNG, mutable buffer and train/eval flag."""
    cpu=torch.get_rng_state().clone()
    cuda=([s.clone() for s in torch.cuda.get_rng_state_all()]
          if torch.cuda.is_available() else None)
    python_rng=random.getstate()
    numpy_rng=np.random.get_state()
    flags=[(layer,layer.training) for layer in model.modules()]
    buffers=[(buffer,buffer.detach().clone()) for buffer in model.buffers()]
    cudnn_flags=(torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic)
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer,original in buffers:
                buffer.copy_(original)
        for layer,training in flags:
            layer.training=training
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.backends.cudnn.benchmark=cudnn_flags[0]
        torch.backends.cudnn.deterministic=cudnn_flags[1]


def mtgnn_bayesian_inference(model, z, graph, name, seed, yc, ys):
    """B-MTGNN's fixed 30-draw MC-dropout forecast and uncertainty."""
    if name!="B-MTGNN":
        raise ValueError("Only B-MTGNN is evaluated in Stage04")
    model.train()
    n_draws=STAGE04_BMTGNN_MC_SAMPLES
    seed_all(seed+100000)
    with torch.inference_mode():
        sampled=np.stack([
            model(z["val_x"],z["val_mask"],graph,
                  z["val_last"],z["val_structure"],z["val_context"])
            .detach().cpu().numpy()[:,0] for _ in range(n_draws)])
    means=(sampled.mean(axis=0)*ys+yc).astype(np.float64)
    if (means.shape!=(z["val_x"].shape[0],NODE_COUNT,3)
            or not np.isfinite(means).all()):
        raise RuntimeError("Invalid B-MTGNN posterior-mean prediction")
    details=dict(
        prediction=means,
        inference_regime="B_MTGNN_MC_DROPOUT",
        inference_draws=int(n_draws))
    if n_draws<2:
        raise RuntimeError("Bayesian inference needs multiple draws")
    stdev=(sampled.std(axis=0,ddof=1)*ys).astype(np.float64)
    # A zero-initialized residual legitimately has zero MC variance.
    # Dropout protocol is checked through the draw count/mode, not by
    # requiring every trained model to retain nonzero predictive variance.
    if not np.isfinite(stdev).all():
        raise RuntimeError("Nonfinite B-MTGNN predictive uncertainty")
    details.update(predictive_mc_std=stdev,predictive_mc_var=stdev**2)
    return details

def fit_three_axis_gnn(panel, data, model_name, seed, device,
                       fixed_epochs: int,
                       prediction_mode: str = "structural",
                       snapshot_epochs=None,checkpoint_epoch=None):
    """Full-Train B-MTGNN fit; no Dev-selected epochs."""
    if model_name!="B-MTGNN":
        raise ValueError("Only B-MTGNN may be trained")
    if (fixed_epochs is None or not isinstance(fixed_epochs,int)
            or not 1<=fixed_epochs<=STAGE04_ROBUST_MAX_EPOCHS):
        raise ValueError("A fixed epoch count is required; no inner validation")
    snapshots=tuple(() if snapshot_epochs is None else snapshot_epochs)
    if (snapshots and (tuple(sorted(set(snapshots)))!=snapshots
                       or snapshots[0]<1 or snapshots[-1]>fixed_epochs
                       or checkpoint_epoch not in snapshots)):
        raise ValueError("Snapshots and reference checkpoint must be in training range")
    seed_all(seed)
    z = {k:torch.as_tensor(data[k],device=device)
        for k in ("train_x","train_mask","train_y","train_last",
                   "train_context","train_structure","val_x","val_mask",
                   "val_last","val_context","val_structure")}
    yc,ys = data["y_center"],data["y_scale"]
    y_scale = torch.as_tensor(ys,device=device)
    # B-MTGNN learns its adjacency from Train gradients; the external
    # argument is unused by the MTGNN core and is not an extra learned graph.
    graph = torch.eye(NODE_COUNT,device=device)
    model = ThreeAxisStructureGNN(
        model_name,data["x_center"],data["x_scale"],yc,ys,
        num_features=data["train_x"].shape[-1],
        nodes=len(panel.nodes),
        prediction_mode=prediction_mode).to(device)
    initialize_calibrated_model(model,data)
    optim = torch.optim.AdamW(
        model.parameters(),lr=FIXED_LEARNING_RATE,weight_decay=1e-4)
    iterations = int(fixed_epochs)
    start = time.perf_counter()
    snapshot_results={}
    saved_state=None
    for epoch in range(1,iterations+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        pred = model(z["train_x"],z["train_mask"],graph,
                     z["train_last"],z["train_structure"],z["train_context"])
        loss = joint_axis_ici_loss(pred,z["train_y"],y_scale)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite common GNN training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.,error_if_nonfinite=True)
        optim.step()
        if epoch in snapshots:
            if epoch==checkpoint_epoch:
                saved_state={k:v.detach().cpu().clone()
                             for k,v in model.state_dict().items()}
            with isolated_snapshot_inference(model):
                snapshot_results[epoch]=mtgnn_bayesian_inference(
                    model,z,graph,model_name,seed,yc,ys)
    if snapshots:
        evaluated=snapshot_results[checkpoint_epoch]
    else:
        with isolated_snapshot_inference(model):
            evaluated=mtgnn_bayesian_inference(
                model,z,graph,model_name,seed,yc,ys)
        saved_state={k:v.detach().cpu().clone()
                     for k,v in model.state_dict().items()}
    prediction=evaluated["prediction"]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite Stage04 GNN structural forecast")
    epoch_predictions=(np.stack([
        snapshot_results[epoch]["prediction"] for epoch in snapshots])
        if snapshots else None)
    total_seconds=time.perf_counter()-start
    nparams=sum(p.numel() for p in model.parameters())
    del model,optim,z
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return dict(**evaluated,state_dict=saved_state,
                snapshot_epochs=snapshots,
                epoch_predictions=epoch_predictions,
                parameters=nparams,seconds=total_seconds,
                best_epoch=checkpoint_epoch if snapshots else fixed_epochs,
                trained_epochs=fixed_epochs)

def naive_full_forecast(panel,model,first=TRAIN_COUNT,last=TOTAL_COUNT):
    if model not in (*EXTRA_NAIVE_BASELINES,"Persistence","SeasonalNaive"):
        raise ValueError(model)
    result=[]
    for t in range(first,last):
        if model=="Persistence":
            point=past_scores(panel,[t])[0]
        elif model=="SeasonalNaive":
            point=origin_lag_scores(panel,[t],4)[0]
        elif model=="HistoricalMean":
            point=np.mean(past_scores(panel,np.arange(1,t+1)),axis=0)
        else:   # no tuning: drift across past-only three-axis input scores
            point=2.*past_scores(panel,[t])[0]-origin_lag_scores(panel,[t],2)[0]
        result.append(point)
    out=np.stack(result)
    if out.shape!=(last-first,NODE_COUNT,3) or not np.isfinite(out).all():
        raise RuntimeError("Naive forecasting alignment failed")
    return out

# ---- The single B-MTGNN benchmark protocol ----

def benchmark_jobs():
    return [(mode,model,seed)
            for mode in STAGE04_OUTPUT_MODES
            for model in MODELS
            for seed in DEFAULT_SEEDS]


def benchmark_contract():
    files=["Data/03_integrated_dataset/raw_train.csv",
           "Data/03_integrated_dataset/raw_val.csv",
           "Data/01_target_data_collection/latest_sources.csv.gz",
           "Results/03_integrated_dataset/target_score_bounds.csv"]
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in files}
    return dict(
        model="B-MTGNN",modes=list(STAGE04_OUTPUT_MODES),
        learning_recipe=LEARNING_RECIPE,
        calibration="MEDIAN_TRAIN_RESIDUAL_PER_INDUSTRY_AXIS",
        train_origins=admissible_train_origins(LOOKBACK).tolist(),
        hidden_width=STAGE04_ACTIVE_WIDTH,learning_rate=FIXED_LEARNING_RATE,
        seeds=list(DEFAULT_SEEDS),train="2006Q1-2019Q4",
        dev="2020Q1-2022Q4",
        bmtgnn_mc_draws=STAGE04_BMTGNN_MC_SAMPLES,
        reference_epoch=STAGE04_ACTIVE_EPOCHS,
        bmtgnn_epoch_stress=list(STAGE04_ROBUST_SNAPSHOTS),
        economic_coefficients=list(STAGE04_FIXED_ALPHA),
        data_sha256=hashes)


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,
                              allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def _write_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp.npz")
    np.savez_compressed(tmp,**arrays)
    tmp.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def job_id(mode,model,seed):
    return f"{mode}__{model}__s{seed}"


def _paths(mode,model,seed):
    stem=job_id(mode,model,seed)
    return (RESULTS/"scores"/f"{stem}.json",
            RESULTS/"predictions"/f"{stem}.npz",
            RESULTS/"models"/f"{stem}.pt")


def make_fold(panel):
    """Use identical full-Train scalers and economically fixed coefficients."""
    n,d,info=economic_components(panel)
    fixed,calibration=select_train_structural_priors(panel,n,d,info)
    structure=component_structural_prior(
        n,d,info,np.arange(len(panel.quarters)),fixed)
    return structure,calibration


def score_forecast(pred,truth):
    """Original three-axis and ICI targets on the Dev partition."""
    err=np.asarray(pred,dtype=float)-np.asarray(truth,dtype=float)
    if err.shape[-2:]!=(NODE_COUNT,3) or not np.isfinite(err).all():
        raise ValueError("Forecast shape/finite mismatch")
    axes=np.abs(err).mean(axis=(0,1))
    results=dict(axis_mae=float(np.abs(err).mean()),
                 ici_mae=float(np.abs(err.mean(-1)).mean()))
    results.update({f"mae_{name}":float(axes[i])
                    for i,name in enumerate(
                        ("profitability","productivity","growth"))})
    for j in range(len(err)//4):
        results[f"year_{j}_mae"]=float(np.abs(err[j*4:j*4+4]).mean())
        results[f"year_{j}_ici_mae"]=float(
            np.abs(err[j*4:j*4+4].mean(-1)).mean())
    return results


def _reference_scores(panel,structure,first,last):
    truths=panel.y[first:last]
    rows=[dict(model="EconomicPrior",mode="reference",seed=None,
               **score_forecast(structure[first:last],truths))]
    for model in NAIVE_MODELS:
        rows.append(dict(model=model,mode="reference",seed=None,
                         **score_forecast(
                             naive_full_forecast(panel,model,first,last),truths)))
    return rows


def _job_dev(panel,structure,mode,model,seed,device,folds,truth,source_sha):
    if model!="B-MTGNN":
        raise ValueError("04 trains only B-MTGNN")
    metric,pred_path,checkpoint=_paths(mode,model,seed)
    if metric.exists() or pred_path.exists() or checkpoint.exists():
        if not all(p.exists() for p in (metric,pred_path,checkpoint)):
            raise RuntimeError(f"Incomplete benchmark job: {job_id(mode,model,seed)}")
        row=_read_json(metric)
        if row["source_sha256"]!=source_sha:
            raise RuntimeError("Benchmark job source changed")
        with np.load(pred_path,allow_pickle=False) as saved:
            if saved["prediction"].shape!=(DEV_COUNT,NODE_COUNT,3):
                raise RuntimeError("Incomplete prediction")
        return row
    scaler,missing=fixed_preprocessing_policy(model)
    key=(scaler,missing)
    if key not in folds:
        folds[key]=prepare_three_axis_fold(
            panel,structure,TRAIN_COUNT,TRAIN_COUNT,TOTAL_COUNT,scaler,missing)
    fold=folds[key]
    metric.parent.mkdir(parents=True,exist_ok=True)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    tic=time.perf_counter()
    trained=fit_three_axis_gnn(
        panel,fold,model,seed,device,STAGE04_ROBUST_MAX_EPOCHS,
        prediction_mode=mode,
        snapshot_epochs=STAGE04_ROBUST_SNAPSHOTS,
        checkpoint_epoch=STAGE04_ACTIVE_EPOCHS)
    temp=checkpoint.with_name(checkpoint.name+".tmp")
    torch.save(trained["state_dict"],temp)
    temp.replace(checkpoint)
    forecast=trained["prediction"]
    _write_npz(
        pred_path,prediction=forecast,
        epochs=np.array(STAGE04_ROBUST_SNAPSHOTS),
        epoch_predictions=trained["epoch_predictions"],
        mc_std=trained["predictive_mc_std"])
    row=dict(
        mode=mode,model=model,seed=seed,reference_epoch=STAGE04_ACTIVE_EPOCHS,
        source_sha256=source_sha,seconds=float(time.perf_counter()-tic),
        **score_forecast(forecast,truth))
    _write_json(metric,row)
    return row


def _read_completed_jobs(source_sha):
    rows=[]
    for mode,model,seed in benchmark_jobs():
        metric,pred_path,checkpoint=_paths(mode,model,seed)
        if not all(p.exists() for p in (metric,pred_path,checkpoint)):
            raise RuntimeError("All 30 B-MTGNN mode/seed fits must be completed")
        row=_read_json(metric)
        if (row.get("source_sha256")!=source_sha
                or (row.get("mode"),row.get("model"),row.get("seed"))!=
                    (mode,model,seed)):
            raise RuntimeError("Stale or inconsistent Stage04 prediction")
        rows.append(row)
    return rows


def _summaries(rows,year_start):
    """Descriptive seed and annual stress, not fake independent p-values."""
    detail=pd.DataFrame(rows)
    summary=[]
    for (model,mode),part in detail.groupby(["model","mode"],sort=True):
        entry=dict(model=model,mode=mode,seed_count=len(part),
                   mae_mean=float(part.axis_mae.mean()),
                   mae_seed_sd=float(part.axis_mae.std(ddof=1)),
                   ici_mae_mean=float(part.ici_mae.mean()),
                   ici_seed_sd=float(part.ici_mae.std(ddof=1)))
        for i,name in enumerate(("profitability","productivity","growth")):
            entry[f"{name}_mae_mean"]=float(part[f"mae_{name}"].mean())
        for j in range(3):
            entry[f"mae_{year_start+j}"]=float(part[f"year_{j}_mae"].mean())
            entry[f"ici_mae_{year_start+j}"]=float(
                part[f"year_{j}_ici_mae"].mean())
        summary.append(entry)
    return detail,pd.DataFrame(summary).sort_values(["model","mode"])


def _epoch_stability(truth):
    """B-MTGNN at epoch 50/75/100; epoch 75 stays fixed for comparison."""
    rows=[]
    for mode in STAGE04_OUTPUT_MODES:
        for seed in DEFAULT_SEEDS:
            _,path,_=_paths(mode,"B-MTGNN",seed)
            with np.load(path,allow_pickle=False) as f:
                if tuple(f["epochs"].tolist())!=STAGE04_ROBUST_SNAPSHOTS:
                    raise RuntimeError("Missing B-MTGNN epoch stress trajectories")
                epochs=f["epoch_predictions"]
            for epoch,forecast in zip(STAGE04_ROBUST_SNAPSHOTS,epochs):
                rows.append(dict(mode=mode,seed=seed,epoch=epoch,
                                 **score_forecast(forecast,truth)))
    return pd.DataFrame(rows)


def _freeze_dev(panel,structure,rows,training_source_sha256):
    if (len(rows)!=30 or
            {(row["mode"],row["model"],row["seed"]) for row in rows}
                !=set(benchmark_jobs())):
        raise RuntimeError("B-MTGNN Dev must contain exactly 30 runs")
    detail,summary=_summaries(rows,2020)
    comparison=pd.concat([
        summary,
        pd.DataFrame(_reference_scores(
            panel,structure,TRAIN_COUNT,TOTAL_COUNT)).rename(
                columns={"axis_mae":"mae_mean","ici_mae":"ici_mae_mean"})
    ],ignore_index=True)
    mode=select_bmtgnn_mode(summary)
    winner=summary.loc[
        (summary.model=="B-MTGNN")&(summary["mode"]==mode)].iloc[0]
    selected=dict(
        model="B-MTGNN",selected_mode=mode,
        selection_rule="LOWEST_DEV_10_SEED_MEAN_THREE_AXIS_MAE_THEN_ICI",
        dev_mean_mae=float(winner["mae_mean"]),
        source_sha256=training_source_sha256,
        seed_count=10)
    detail.to_csv(RESULTS/"dev_scores.csv",index=False)
    comparison.to_csv(RESULTS/"dev_summary.csv",index=False)
    _epoch_stability(panel.y[TRAIN_COUNT:TOTAL_COUNT]).to_csv(
        RESULTS/"dev_bmtgnn_epoch_stability.csv",index=False)
    _write_json(RESULTS/"dev_selection.json",selected)
    return selected


def select_bmtgnn_mode(summary):
    """Compare only B-MTGNN's three Dev target formulations."""
    b=summary.loc[summary["model"]=="B-MTGNN"]
    if set(b["mode"])!=set(STAGE04_OUTPUT_MODES):
        raise ValueError("All three B-MTGNN methods must be scored on Dev")
    return str(b.sort_values(
        ["mae_mean","ici_mae_mean","mode"]).iloc[0]["mode"])


def summarize_development():
    """Build ONLY B-MTGNN summaries from the 30 source-attested Dev jobs."""
    manifest=RESULTS/"run_manifest.json"
    if not manifest.exists():
        raise RuntimeError("B-MTGNN Dev jobs do not exist")
    status=_read_json(manifest)
    if (status["status"]!="DEV_FROZEN"
            or status["contract"]!=benchmark_contract()
            or status["completed_jobs"]!=30
            or status["total_jobs"]!=30):
        raise RuntimeError("Complete the B-MTGNN-only Dev protocol first")
    training_sha=status["training_source_sha256"]
    rows=_read_completed_jobs(training_sha)
    panel=load_panel(ROOT)
    structure,_=make_fold(panel)
    selection=_freeze_dev(panel,structure,rows,training_sha)
    if selection["selected_mode"]!=status["selected_mode"]:
        raise RuntimeError("Frozen B-MTGNN output formulation changed")
    return selection


def reset_stage_results(path):
    """Clear one existing result directory ONLY on an explicit --restart.

    Never relabel old learning results with a new source signature, move
    folders, or delete upstream data. Calling this is an execution action;
    changing the source or importing a module cannot trigger it.
    """
    names=("04_bmtgnn_benchmark","05_input_lookback","06_preprocessing",
           "07_correlated_feature_removal","08_feature_selection","09_hyperparameter_tuning")
    allowed={ROOT/"Results"/name for name in names}
    path=Path(path)
    if path not in allowed or path.resolve()!=path:
        raise ValueError("Restart is restricted to the six existing stage result directories")
    if path.exists():
        shutil.rmtree(path)


def run_development(limit=None,device="cpu",restart=False):
    """Train once and freeze the B-MTGNN mode using Dev only."""
    jobs=benchmark_jobs()
    assert len(jobs)==30 and len(DEFAULT_SEEDS)==10
    contract=benchmark_contract()
    current_sha=hashlib.sha256(
        (ROOT/"Sources/04_bmtgnn_benchmark.py").read_bytes()).hexdigest()
    if restart:
        reset_stage_results(RESULTS)
    manifest=RESULTS/"run_manifest.json"
    if manifest.is_file():
        status=_read_json(manifest)
        if status["contract"]!=contract:
            raise RuntimeError("Current Stage04 protocol differs from stored run")
        digest=status["training_source_sha256"]
        if digest!=current_sha:
            raise RuntimeError("Stage04 source changed; explicitly --restart this stage")
        if status["status"]=="DEV_FROZEN":
            _read_completed_jobs(digest)
            return status
    else:
        digest=current_sha
        status=dict(status="DEV_RUNNING",contract=contract,
                    training_source_sha256=digest,
                    completed_jobs=0,total_jobs=len(jobs))
        _write_json(manifest,status)
    panel=load_panel(ROOT)
    structure,_=make_fold(panel)
    truth=panel.y[TRAIN_COUNT:TOTAL_COUNT]
    folds={}
    done=0
    torch.set_num_threads(4)
    for mode,model,seed in jobs:
        if _paths(mode,model,seed)[0].is_file():
            _job_dev(panel,structure,mode,model,seed,device,folds,truth,digest)
            continue
        if limit is not None and done>=limit:
            break
        _job_dev(panel,structure,mode,model,seed,device,folds,truth,digest)
        done+=1
        if done%5==0:
            print(f"DEV {done}/{len(jobs)} in this run",flush=True)
    completed=sum(_paths(*job)[0].exists() for job in jobs)
    status["completed_jobs"]=completed
    if completed==len(jobs):
        selected=_freeze_dev(
            panel,structure,_read_completed_jobs(digest),digest)
        status.update(status="DEV_FROZEN",selected_mode=selected["selected_mode"],
                      frozen_utc=datetime.now(timezone.utc).isoformat())
    _write_json(manifest,status)
    return status


def plot_development():
    """Regenerate original 16-panel hierarchy PLUS B-MTGNN robustness plots.

    Only the 30 frozen B-MTGNN Dev fits and non-trained reference predictions
    are used. There is no estimator ranking or held-out-split access here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest=_read_json(RESULTS/"run_manifest.json")
    if (manifest.get("status")!="DEV_FROZEN" or
            manifest.get("completed_jobs")!=30 or
            manifest.get("total_jobs")!=30):
        raise RuntimeError("All 30 B-MTGNN Dev fits must be frozen to plot")

    detail=pd.read_csv(RESULTS/"dev_scores.csv")
    summary=pd.read_csv(RESULTS/"dev_summary.csv")
    epochs=pd.read_csv(RESULTS/"dev_bmtgnn_epoch_stability.csv")
    jobs=set(zip(detail["mode"],detail["model"],detail["seed"].astype(int)))
    stresses=set(zip(epochs["mode"],epochs["seed"].astype(int),
                     epochs["epoch"].astype(int)))
    expected_stresses={(mode,seed,ep)
        for mode in STAGE04_OUTPUT_MODES
        for seed in DEFAULT_SEEDS for ep in STAGE04_ROBUST_SNAPSHOTS}
    if (len(detail)!=30 or jobs!=set(benchmark_jobs())
            or len(epochs)!=90 or stresses!=expected_stresses
            or set(summary.model)!={"B-MTGNN","EconomicPrior",*NAIVE_MODELS}):
        raise RuntimeError("B-MTGNN-only Dev records are incomplete")

    panel=load_panel(ROOT)
    truth=panel.y[TRAIN_COUNT:TOTAL_COUNT]
    previous=past_scores(panel,np.arange(TRAIN_COUNT,TOTAL_COUNT))
    economic,_=make_fold(panel)
    prior=economic[TRAIN_COUNT:TOTAL_COUNT]
    baseline=score_forecast(prior,truth)
    naive=naive_full_forecast(
        panel,"SeasonalNaive",TRAIN_COUNT,TOTAL_COUNT)
    reference_row=summary.loc[summary.model=="EconomicPrior"].iloc[0]
    if not np.isclose(baseline["axis_mae"],reference_row.mae_mean,atol=1e-8):
        raise RuntimeError("Economic prior differs from frozen Dev summary")

    modes=STAGE04_OUTPUT_MODES
    axes_names=("Profitability","Productivity","Growth","ICI")
    quarters=panel.quarters[TRAIN_COUNT:TOTAL_COUNT]
    years=(2020,2021,2022)
    output=RESULTS/"plots"
    output.mkdir(parents=True,exist_ok=True)
    generated=[]
    predictions={}
    uncertainty={}
    for mode in modes:
        scores=[]
        stds=[]
        for seed in DEFAULT_SEEDS:
            _,path,_=_paths(mode,"B-MTGNN",seed)
            with np.load(path,allow_pickle=False) as values:
                pred=values["prediction"].copy()
                mc_std=values["mc_std"].copy()
                eps=values["epoch_predictions"]
                if (pred.shape!=truth.shape or mc_std.shape!=truth.shape
                        or eps.shape!=(3,*truth.shape) or
                        tuple(values["epochs"].tolist())!=
                            STAGE04_ROBUST_SNAPSHOTS or
                        not np.allclose(eps[1],pred,rtol=0,atol=0)):
                    raise RuntimeError("Incomplete B-MTGNN Dev snapshots")
            row=detail[(detail["mode"]==mode)&(detail["seed"]==seed)]
            if (len(row)!=1 or not np.isclose(
                    np.abs(pred-truth).mean(),row.iloc[0].axis_mae,atol=1e-8)):
                raise RuntimeError("Saved predictions do not match Dev scores")
            scores.append(pred)
            stds.append(mc_std)
        predictions[mode]=np.stack(scores)
        uncertainty[mode]=np.stack(stds)

    def save(fig,name):
        path=output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.tight_layout()
        fig.savefig(path,dpi=160,bbox_inches="tight")
        plt.close(fig)
        generated.append(path.relative_to(output).as_posix())

    def values(data,axis):
        return (data.mean(-1) if axis==3 else data[...,axis]).mean(1)

    def axis_ici_mae(data):
        error=data-truth
        return np.r_[np.abs(error).mean((0,1)),
                     np.abs(error.mean(-1)).mean()]

    def reference_year(year,metric):
        j=year-2020
        return baseline[f"year_{j}_{metric}"]

    model=summary[summary.model=="B-MTGNN"].set_index("mode").loc[list(modes)]

    # Original four root-level panels, with B-MTGNN instead of GNN ensembles.
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(3)
    for j,(label,errs) in enumerate((
        ("B-MTGNN",[model.loc[m,"mae_mean"] for m in modes]),
        ("Economic prior",[baseline["axis_mae"]]*3),
        ("Persistence",[np.abs(previous-truth).mean()]*3))):
        ax.bar(x+(j-1)*.25,errs,.25,label=label)
    ax.set_xticks(x,[m.title() for m in modes])
    ax.set(ylabel="Dev three-axis MAE",title="Target prediction formulations")
    ax.grid(axis="y",alpha=.2)
    ax.legend()
    save(fig,"01_three_modes_model_selection.png")

    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    for j,ax in enumerate(axes.flat):
        ax.plot(quarters,values(truth,j),"k--",lw=2.4,label="Actual")
        ax.plot(quarters,values(previous,j),":",color="gray",
                lw=1.8,label="Persistence")
        for mode in modes:
            ax.plot(quarters,values(predictions[mode].mean(0),j),
                    lw=1.9,label=f"{mode.title()} B-MTGNN")
        ax.set(title=axes_names[j],ylabel="59-industry mean score")
        ax.tick_params(axis="x",rotation=35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Observed vs B-MTGNN target methods and Persistence")
    save(fig,"02_actual_vs_modes_with_persistence.png")

    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(quarters,(truth-previous)[...,j].mean(1),
                "k--",lw=2.4,label="Actual change")
        ax.plot(quarters,np.zeros(12),":",color="gray",
                lw=1.6,label="Persistence (zero)")
        for mode in modes:
            change=(predictions[mode].mean(0)-previous)[...,j].mean(1)
            ax.plot(quarters,change,lw=1.9,label=f"{mode.title()} B-MTGNN")
        ax.set(title=f"{axes_names[j]} next-quarter change",
               ylabel="Mean original-score change")
        ax.tick_params(axis="x",rotation=35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig,"03_change_vs_modes_with_persistence.png")

    fig,ax=plt.subplots(figsize=(11,5))
    for mode in modes:
        grouped=epochs[epochs["mode"]==mode].groupby("epoch").axis_mae.mean()
        ax.plot(grouped.index,grouped.values,".-",label=mode.title())
        ax.scatter([75],[grouped.loc[75]],s=90,marker="*")
    ax.set(xlabel="Training Epoch (not model count)",
           ylabel="Dev three-axis MAE",
           title="B-MTGNN fixed-epoch robustness")
    ax.set_xticks(STAGE04_ROBUST_SNAPSHOTS)
    ax.legend()
    ax.grid(alpha=.2)
    save(fig,"04_gnn_ensemble_k_by_mode.png")

    # Identical original four-plot hierarchy within each target formulation.
    for mode in modes:
        folder=mode.capitalize()
        rows=detail[detail["mode"]==mode].sort_values("seed")
        mean_pred=predictions[mode].mean(0)
        fig,axes=plt.subplots(1,2,figsize=(18,15))
        for ax,metric,title,refval in (
            (axes[0],"axis_mae","Three-axis MAE",baseline["axis_mae"]),
            (axes[1],"ici_mae","ICI MAE",baseline["ici_mae"])):
            ax.barh([str(seed) for seed in rows.seed],rows[metric],
                    color="tab:blue")
            ax.axvline(refval,color="black",ls=":",lw=1.5,
                       label="Economic prior")
            ax.invert_yaxis()
            ax.set(title=title,xlabel="Dev MAE",ylabel="B-MTGNN seed")
            ax.grid(axis="x",alpha=.2)
            ax.legend()
        fig.suptitle(f"{mode.title()}: ten-seed B-MTGNN robustness")
        save(fig,f"{folder}/01_model_comparison.png")

        fig,axes=plt.subplots(1,2,figsize=(14,5))
        subset=epochs[epochs["mode"]==mode]
        for ax,metric,title in (
            (axes[0],"axis_mae","Three-axis MAE"),
            (axes[1],"ici_mae","ICI MAE")):
            grp=subset.groupby("epoch")[metric]
            mean,std=grp.mean(),grp.std()
            ax.errorbar(mean.index,mean.values,yerr=std.values,fmt=".-",
                        capsize=4,label="Seed mean ± SD")
            ax.scatter([75],[mean.loc[75]],marker="*",color="tab:red",
                       s=180,label="Fixed Epoch 75",zorder=3)
            ax.set(xlabel="Training Epoch",ylabel="Dev MAE",title=title)
            ax.set_xticks(STAGE04_ROBUST_SNAPSHOTS)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{mode.title()}: Epoch robustness")
        save(fig,f"{folder}/02_ensemble_selection.png")

        fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
        for j,ax in enumerate(axes.flat):
            ax.plot(quarters,values(truth,j),"k--",lw=2.2,label="Actual")
            ax.plot(quarters,values(previous,j),":",color="gray",
                    lw=1.8,label="Persistence")
            ax.plot(quarters,values(mean_pred,j),lw=1.8,label="B-MTGNN")
            ax.plot(quarters,values(prior,j),"-.",
                    lw=1.4,label="Economic prior")
            ax.set(title=axes_names[j],ylabel="59-industry mean score")
            ax.tick_params(axis="x",rotation=40)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{mode.title()}: observed, forecasts, persistence")
        save(fig,f"{folder}/03_actual_vs_predictions_with_persistence.png")

        fig,ax=plt.subplots(figsize=(11,5))
        comparisons=(
            ("Persistence",axis_ici_mae(previous)),
            ("B-MTGNN",axis_ici_mae(mean_pred)),
            ("Economic prior",axis_ici_mae(prior)),
            ("Seasonal naive",axis_ici_mae(naive)))
        x=np.arange(4)
        for j,(title,vals) in enumerate(comparisons):
            ax.bar(x+(j-1.5)*.19,vals,.19,label=title)
        ax.set_xticks(x,axes_names)
        ax.set(ylabel="Dev MAE",title=f"{mode.title()}: Three axes and ICI")
        ax.grid(axis="y",alpha=.2)
        ax.legend(fontsize=9)
        save(fig,f"{folder}/04_axis_ici_comparison.png")

    # Six new dedicated robustness plots.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title,ref in (
        (axes[0],"axis_mae","Three-axis MAE",baseline["axis_mae"]),
        (axes[1],"ici_mae","ICI MAE",baseline["ici_mae"])):
        groups=[detail[detail["mode"]==m][metric].to_numpy() for m in modes]
        ax.boxplot(groups,positions=np.arange(1,4),widths=.55)
        for j,group in enumerate(groups,1):
            ax.scatter(np.full(len(group),j),group,s=20,alpha=.6)
        ax.axhline(ref,ls="--",color="tab:orange",label="Economic prior")
        ax.set_xticks(np.arange(1,4),[m.title() for m in modes])
        ax.set(title=title,ylabel="Dev error, ten training seeds")
        ax.grid(axis="y",alpha=.2)
        ax.legend()
    fig.suptitle("Seed robustness | B-MTGNN")
    save(fig,"Robustness/01_seed_distribution.png")

    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,mode in zip(axes,modes):
        pivot=epochs[epochs["mode"]==mode].pivot(
            index="seed",columns="epoch",values="axis_mae"
        ).loc[list(DEFAULT_SEEDS),list(STAGE04_ROBUST_SNAPSHOTS)]
        graphic=ax.imshow(pivot.to_numpy(),aspect="auto",cmap="viridis")
        ax.set_xticks(np.arange(3),list(STAGE04_ROBUST_SNAPSHOTS))
        ax.set_yticks(np.arange(10),list(DEFAULT_SEEDS))
        ax.set(title=mode.title(),xlabel="Epoch",ylabel="Seed")
        fig.colorbar(graphic,ax=ax,label="Three-axis MAE")
    fig.suptitle("Seed × Epoch robustness")
    save(fig,"Robustness/02_seed_epoch_heatmap.png")

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,kind,title in ((axes[0],"mae","Three-axis MAE"),
                          (axes[1],"ici_mae","ICI MAE")):
        data=np.asarray([
            [100.*(detail[detail["mode"]==mode][
                f"year_{j}_{kind}"].max()/
                 baseline[f"year_{j}_{kind}"]-1.)
             for j in range(3)] for mode in modes])
        bound=max(1.,float(np.abs(data).max()))
        picture=ax.imshow(data,aspect="auto",cmap="coolwarm",
                          vmin=-bound,vmax=bound)
        ax.set_xticks(np.arange(3),years)
        ax.set_yticks(np.arange(3),[m.title() for m in modes])
        ax.set(title=title,xlabel="Dev year",ylabel="Method")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{data[i,j]:+.1f}%",ha="center",va="center")
        fig.colorbar(picture,ax=ax,label="% vs economic prior")
    fig.suptitle("Worst observed seed/year regret (Dev-only stress)")
    save(fig,"Robustness/03_worst_year_seed_regret.png")

    industry=[("Economic prior",np.abs(prior-truth).mean((0,2)),
               np.abs((prior-truth).mean(-1)).mean(0))]
    for mode in modes:
        err=predictions[mode]-truth[None]
        industry.append((mode.title(),np.abs(err).mean((0,1,3)),
                         np.abs(err.mean(-1)).mean((0,1))))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,index,title in ((axes[0],1,"Three-axis MAE"),
                           (axes[1],2,"ICI MAE")):
        ax.boxplot([row[index] for row in industry])
        ax.set_xticks(np.arange(1,5),[row[0] for row in industry],
                      rotation=15)
        ax.set(title=title,ylabel="Per-industry error, 59 industries")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Industry variability (correlated units)")
    save(fig,"Robustness/04_industry_error_dispersion.png")

    fig,ax=plt.subplots(figsize=(10,6))
    for mode in modes:
        std=uncertainty[mode].ravel()
        error=np.abs(predictions[mode]-truth[None]).ravel()
        limits=np.quantile(std,np.linspace(0,1,11))
        bins=np.searchsorted(limits[1:-1],std,side="right")
        x=np.asarray([std[bins==j].mean() for j in range(10)])
        y=np.asarray([error[bins==j].mean() for j in range(10)])
        ax.plot(x,y,marker="o",lw=1.8,label=mode.title())
    ax.set(xlabel="MC dropout dispersion (decile mean)",
           ylabel="Absolute error (decile mean)",
           title="MC uncertainty–error association, not calibrated coverage")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,"Robustness/05_mc_uncertainty_error_relation.png")

    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for ax,mode in zip(axes,modes):
        yearly=[detail[detail["mode"]==mode][
            f"year_{j}_mae"].to_numpy() for j in range(3)]
        ax.boxplot(yearly,positions=np.arange(3),widths=.55)
        ax.plot(np.arange(3),[baseline[f"year_{j}_mae"]
                              for j in range(3)],"--o",color="tab:orange",
                label="Economic prior")
        ax.set(title=mode.title(),ylabel="10-seed annual MAE")
        ax.grid(axis="y",alpha=.2)
        ax.legend()
    axes[-1].set_xticks(np.arange(3),years)
    axes[-1].set_xlabel("Dev year")
    fig.suptitle("Annual variation × seed stability")
    save(fig,"Robustness/06_year_seed_stability.png")

    # Four new plots comparing target formulations, not model families.
    fig,ax=plt.subplots(figsize=(12,6))
    x=np.arange(4)
    for j,mode in enumerate(modes):
        ax.bar(x+(j-1.5)*.19,
               axis_ici_mae(predictions[mode].mean(0)),
               .19,label=mode.title())
    ax.bar(x+1.5*.19,axis_ici_mae(prior),.19,
           label="Economic prior",color="tab:gray")
    ax.set_xticks(x,axes_names)
    ax.set(ylabel="Original-score Dev MAE",
           title="Target formulations: three axes and ICI")
    ax.grid(axis="y",alpha=.2)
    ax.legend()
    save(fig,"Methodology/01_target_axis_methods.png")

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in ((axes[0],"mae","Three-axis MAE"),
                            (axes[1],"ici_mae","ICI MAE")):
        ax.plot(years,[baseline[f"year_{j}_{metric}"]
                       for j in range(3)],"--o",lw=2,
                label="Economic prior")
        for mode in modes:
            rows=detail[detail["mode"]==mode]
            ax.plot(years,[rows[f"year_{j}_{metric}"].mean()
                           for j in range(3)],marker="o",label=mode.title())
        ax.set_xticks(years)
        ax.set(title=title,xlabel="Dev year",ylabel="MAE")
        ax.grid(alpha=.2)
    axes[1].legend()
    fig.suptitle("Annual target formulation comparison")
    save(fig,"Methodology/02_annual_methods.png")

    fig,ax=plt.subplots(figsize=(11,5))
    for seed in DEFAULT_SEEDS:
        matched=detail[detail.seed==seed].set_index("mode").loc[list(modes)]
        ax.plot(modes,matched.axis_mae,marker="o",lw=1,alpha=.5,
                label=f"Seed {seed}")
    ax.axhline(baseline["axis_mae"],color="tab:orange",ls="--",
               label="Economic prior")
    ax.set(title="Paired ten-seed target methodology comparison",
           ylabel="Three-axis Dev MAE")
    ax.grid(axis="y",alpha=.2)
    ax.legend(ncol=3,fontsize=7)
    save(fig,"Methodology/03_paired_seed_methods.png")

    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    for ax,kind,title in ((axes[0],"mae","Three-axis MAE"),
                          (axes[1],"ici","ICI MAE")):
        err=prior-truth
        ref=(np.abs(err).mean((1,2)) if kind=="mae"
             else np.abs(err.mean(-1)).mean(1))
        ax.plot(quarters,ref,ls="--",lw=2,label="Economic prior")
        for mode in modes:
            diff=predictions[mode]-truth[None]
            score=(np.abs(diff).mean((0,2,3)) if kind=="mae"
                   else np.abs(diff.mean(-1)).mean((0,2)))
            ax.plot(quarters,score,marker=".",lw=1.6,
                    label=mode.title())
        ax.set(title=title,ylabel="Quarterly Dev error")
        ax.tick_params(axis="x",rotation=35)
        ax.grid(alpha=.2)
    axes[1].legend(ncol=4)
    fig.suptitle("Quarterly target formulation error")
    save(fig,"Methodology/04_quarterly_errors.png")

    if len(generated)!=26 or len(set(generated))!=26:
        raise RuntimeError("Expected 16 legacy layouts plus 10 new plots")
    return generated

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("dev","plots","status","dry-run"),
                        default="dev")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--restart",action="store_true",
                        help="Explicitly replace this stage's existing results; no folder rename")
    parser.add_argument("--device",choices=("cpu","cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    if args.restart and args.phase!="dev":
        parser.error("--restart is only valid for --phase dev")
    if args.limit is not None and args.limit<1:
        parser.error("--limit must be positive")
    if args.phase=="dry-run":
        print("B-MTGNN ONLY; ten seeds",DEFAULT_SEEDS)
        print("MODELS",MODELS,"METHODS",STAGE04_OUTPUT_MODES)
        print("DEV JOBS",len(benchmark_jobs()),
              "B-MTGNN EPOCHS",STAGE04_ROBUST_SNAPSHOTS)
        print("Dev-only. No later-split evaluation entry point.")
    elif args.phase=="status":
        path=RESULTS/"run_manifest.json"
        print(_read_json(path) if path.exists() else {"status":"NOT_STARTED"})
    elif args.phase=="dev":
        print(run_development(args.limit,args.device,args.restart)["status"])
    else:
        print("\n".join(plot_development()))


if __name__=="__main__":
    main()
