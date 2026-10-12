#!/usr/bin/env python3
"""PRISM 04 - ONE canonical three-mode forecasting benchmark.

Forecast the EXACT SAME original Stage03 Target at t (profitability,
productivity, growth) for Direct / last-Target Residual / economic-Structural
Residual; targets, input horizons and original ICI are NEVER redefined.

FIXED PREPROCESSING PHILOSOPHY (STAGE 04 IS A MODEL BENCHMARK):
- Preprocessing is NOT another model-selection grid here. Each family has
  ONE scientifically reasonable, predefined input policy; it stays identical
  across Direct / Residual / Structural, all seeds, and Train / Dev.
- GNNs, temporal networks, linear models, SVR/KNN and MLP: Standard
  normalization fitted on observed TRAIN values only. Missing values are
  replaced by RAW ZERO before normalization, with a separate binary
  MISSINGNESS MASK. NEVER use a Train-mean imputation. A raw-zero entry
  generally does NOT equal zero after Standard normalization.
- Tree models supporting missing values (LightGBM, XGBoost, CatBoost,
  RandomForest, ExtraTrees, HistGradientBoosting): native unscaled values
  and genuine NaNs. Classical GradientBoosting has no native NaN support;
  it uses unscaled raw-zero replacement plus a missingness mask instead.
- Simple historical naive benchmarks use their original definitions.
- NO Dev-based tuning/selection of preprocessing in Stage04. A LATER
  pipeline stage (06) may compare normalization, missingness and outlier
  treatments and re-evaluate promising models under new joint policies.

WHY THE ECONOMIC PRIOR USES FIXED ALPHA (0.25, 0.1, 0.25):
- Alpha controls how strongly each axis extrapolates its recent quarterly
  numerator/denominator change. Larger alpha reacts more aggressively;
  smaller alpha provides a more conservative, stable forecasting baseline.
- Profitability/productivity/growth use alpha=0.25/0.1/0.25 respectively.
  On ALL 2006-2019 TRAIN years (56 quarters, 14 years), evaluate the
  same predefined alpha grid, including the early 2006-2008 quarters,
  using only their already-completed 2005 historical components.
  calculate per-year MAE, and select the SMALLEST alpha within one paired-
  by-year standard error of the best mean MAE for each axis.
- This deliberately favors robustness over tiny in-sample MAE differences.
  No 2018-2019 TRAIN subset is withheld for alpha: the complete TRAIN period
  determines the fixed coefficient. 2020-2022 Dev is evaluation-only.
- The separate, Train-only 2018-2019 neural-network epoch check does NOT
  select alpha; it exists to avoid choosing epochs using Dev outcomes.
- Alpha is frozen and identical for inner and full Train experiments.
  No adaptive alpha or performance-driven retuning happens in Stage04.
- PLANNED LATER: compare adaptive/time-varying alpha and optimized economic
  priors as DISTINCT downstream pipeline methods, evaluated against this
  unchanged Stage04 baseline with strictly past-only information.

Each model is run with exactly one predeclared preprocessing policy, with
five seeds for trained models, identically across all three output modes.
KEEP ALL 41 GNNs and the six existing temporal/DL comparators. Classical
ML is now deliberately restricted to three representative, already-computed
baselines: Ridge (linear), LightGBM (boosting), ExtraTrees (randomized
tree ensemble). The removed overlapping ML candidates were expensive to
train and would obscure the intended GNN-centered comparison. No model was
selected or removed using Dev ranking: this is a family-coverage/budget
decision. Never drop the fixed GNN-only ensemble search.
GNN ensembles are GNN-only, with original correlation-aware greedy and
industry-paired 1-SE selection, de-duplicated by exact prediction identity.
IMPORTANT: MTGNN vs B-MTGNN are NOT different trained graph backbones.
The published B-MTGNN extends MTGNN by KEEPING dropout active at inference
and averaging Monte Carlo draws to approximate predictive uncertainty.
MTGNN instead predicts deterministically with dropout disabled.
Stage04 trains/evaluates 04's same MTGNN backbone and loss in both arms.
The 30-draw Bayesian mean and per-cell predictive dispersion are reported
ONLY for B-MTGNN, and MTGNN uses one eval-mode forecast. Older identical
B-MTGNN/MTGNN outputs were an invalid duplicate caused by applying 30
stochastic train-mode forecasts to both models.
All preprocessing/epochs are Train-fitted. Dev 2020-2022 for model comparison.
TEST 2023-2025 is never opened. Latest revised official values are the
agreed, authoritative retrospective reference; do not restore release-date
vintages or confuse this policy with forecast-origin lookahead checks.

Run/resume: python -u Sources/04_gnn_benchmark.py --time-budget-minutes 10000
Preview: --dry-run. Progress: --full-status.
Saved-result figures only: --plots-only (does not train).
Results: Results/04_gnn_benchmark/
"""


from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import hashlib
import json
import math
import random
import time
import traceback
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# -------- REGISTRY / FROZEN INPUTS --------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results/04_gnn_benchmark"
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]
FIXED_LEARNING_RATE = .001
FIXED_BOOSTING = dict(n_estimators=60, num_leaves=15, learning_rate=.05)
TARGETS = ["target_profitability", "target_productivity", "target_growth"]
IDS = ["quarter", "node_key", "node_name"]
LOOKBACK, HORIZON = 12, 1
NODE_COUNT = 59
TRAIN_COUNT = 56                 # 2006Q1-2019Q4; these are quarters, NOT nodes
DEV_COUNT = 12                   # 2020Q1-2022Q4
TOTAL_COUNT = TRAIN_COUNT + DEV_COUNT
SPATIAL = (
    "GCN", "GraphSAGE", "GraphConv", "GIN", "GAT", "GATv2",
    "GraphTransformer", "ChebNet", "SGC", "TAGCN", "APPNP", "GCNII",
    "ARMA", "AGNN", "GatedGraph",
)
SPECIAL = ("MTGNN", "B-MTGNN", "GraphWaveNet", "AGCRN", "DCRNN", "STGCN",
           "T-GCN", "GConvGRU", "GConvLSTM", "EvolveGCN-H-GRU", "EvolveGCN-O-GRU")
GNN_MODELS = list(SPECIAL) + [f"{s}-{t}" for t in ("GRU", "TCN") for s in SPATIAL]
NEURAL_BASELINES = ("GRU", "TCN", "Transformer")
BASELINES = ("Persistence", "SeasonalNaive", "LightGBM", *NEURAL_BASELINES)
EXTRA_NEURAL_BASELINES = ("LSTM", "DLinear", "NBEATS")
EXTRA_ML_BASELINES = ("Ridge", "ExtraTrees")
EXTRA_NAIVE_BASELINES = ("HistoricalMean", "LinearDrift")
OPTIONAL_ML_BASELINES = ()
# One fixed preprocessing assignment per model family, never a Dev-tuned
# grid. The only representative traditional ML comparators are:
# linear Ridge, LightGBM gradient boosting, ExtraTrees randomized bagging.
# Both retained tree families support raw NaN; no preprocessing search.
NATIVE_NAN_TREES = ("LightGBM", "ExtraTrees")
ZERO_MASK_TREES = ()
NAIVE_MODELS = ("Persistence", "SeasonalNaive", *EXTRA_NAIVE_BASELINES)
STAGE04_PREPROCESSING_POLICY = "FIXED_MODEL_FAMILY_NO_DEV_POLICY_TUNING"
STAGE04_BMTGNN_MC_SAMPLES = 30
STAGE04_BMTGNN_INNER_MC_SAMPLES = 10
STAGE04_MTGNN_DETERMINISTIC_SAMPLES = 1
STAGE04_ACTIVE_EPOCHS = 75
STAGE04_ACTIVE_PATIENCE = 15
STAGE04_ACTIVE_WIDTH = 16
STAGE04_ACTIVE_SEEDS = [2026, 2027, 2028, 2029, 2030]
# Stage04 is the deliberately NON-adaptive economic baseline.
# These coefficients were frozen using ALL annual blocks 2006-2019:
# smallest candidate within one paired-by-year SE of the lowest mean MAE.
# The distinct 2018-2019 neural epoch check and 2020-2022 Dev NEVER
# readjust alpha; all Train years do participate in its original selection.
# Later stages may compare adaptive/optimized alternatives separately.
STAGE04_ACTIVE_ALPHA_GRID = (0., .1, .2, .25, .3, .5, .75, 1.)
STAGE04_FIXED_ALPHA = (0.25, 0.1, 0.25)  # profitability, productivity, growth
STAGE04_ALPHA_CALIBRATION_END = TRAIN_COUNT  # 2019Q4, ALL Train quarters
STAGE04_ALPHA_POLICY = "FROZEN_TRAIN_2006_2019_YEAR_PAIRED_ONE_SE"
STAGE04_ACTIVE_EPOCH_CUT = 48  # fit to 2017Q4, holdout 2018Q1–2019Q4
STAGE04_OUTPUT_MODES = ("direct", "residual", "structural")
STAGE04_COMPONENT_SOURCE = ROOT / "Data/01_target_data_collection/latest_sources.csv.gz"
STAGE04_TARGET_BOUNDS = ROOT / "Results/03_integrated_dataset/target_score_bounds.csv"

# ------- GRAPH / LEARNERS / THREE HEADS / RUNNER -------

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
    """No raw_test.csv read, even for scaling, graphs, or model selection."""
    tables, hashes = [], {}
    for name, expected in (("train", ("2006Q1", "2019Q4", TRAIN_COUNT * NODE_COUNT)),
                            ("val", ("2020Q1", "2022Q4", DEV_COUNT * NODE_COUNT))):
        path = root / "Data/03_integrated_dataset" / f"raw_{name}.csv"
        hashes[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, dtype={"quarter": str, "node_key": str}, low_memory=False)
        if (frame.quarter.min(), frame.quarter.max(), len(frame)) != expected:
            raise ValueError(f"Unexpected Stage 03 {name} partition")
        tables.append(frame)
    if tables[0].columns.tolist() != tables[1].columns.tolist():
        raise ValueError("Train/Validation schema mismatch")
    frame = pd.concat(tables, ignore_index=True)
    features = [c for c in frame if c not in IDS + TARGETS + ["target_ici"]]
    if features[:3] != ["history_profitability", "history_productivity", "history_growth"]:
        raise ValueError("Three previous-quarter target features must be first")
    nodes = sorted(frame.node_key.unique())
    quarters = pd.period_range("2006Q1", "2022Q4", freq="Q").astype(str).tolist()
    engineered_count = len(features) - 115
    if (len(nodes) != NODE_COUNT or engineered_count != 42
            or any(not c.startswith(("history_", "targethist_")) for c in features[:engineered_count])
            or any(c.startswith(("history_", "targethist_")) for c in features[engineered_count:])
            or frame.duplicated(["quarter", "node_key"]).any()):
        raise ValueError("Require 59 industries, 42 Target-history and 115 original predictors")
    index = pd.MultiIndex.from_product([quarters, nodes], names=["quarter", "node_key"])
    frame = frame.set_index(["quarter", "node_key"]).reindex(index)
    y = frame[TARGETS].to_numpy(float).reshape(TOTAL_COUNT, NODE_COUNT, 3)
    x = frame[features].to_numpy(float).reshape(TOTAL_COUNT, NODE_COUNT, len(features))
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError("Nonfinite targets or infinite predictors")
    if not np.allclose(frame.target_ici.to_numpy(), y.mean(-1).reshape(-1)):
        raise ValueError("Stage 03 ICI identity mismatch")
    names = frame["node_name"].groupby(level="node_key").first().reindex(nodes).tolist()
    return Panel(x, y, quarters, nodes, names, features, TRAIN_COUNT, hashes)

def fit_normalizer(x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit only observed Train entries, deterministic across memory layouts."""
    if kind not in ("standard", "robust"):
        raise ValueError(kind)
    # 2/3D ndarray views vs copies may sum in a different floating-point
    # order. Flatten deterministically first, so poisoning unused Dev rows
    # cannot appear to perturb Train normalization at the 1e-16 scale.
    values=np.ascontiguousarray(
        np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "standard":
            center, scale = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
        else:
            center = np.nanmedian(values, axis=0)
            q = np.nanquantile(values, [.25, .75], axis=0)
            scale = q[1] - q[0]
    center = np.where(np.isfinite(center), center, 0.)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-8), scale, 1.)
    return center, scale

def fixed_preprocessing_policy(model: str) -> tuple[str, str]:
    """Precisely one policy (scaler, missing) per registered model.

    "mask" means RAW-zero imputation plus a separate missingness indicator.
    For native-NaN models, "none/native" means original values and NaNs
    without any normalization, filling, or indicator expansion.
    """
    if model in NAIVE_MODELS:
        return "none", "none"
    if model in NATIVE_NAN_TREES:
        return "none", "native"
    if model in ZERO_MASK_TREES:
        return "none", "mask"
    if model in (*GNN_MODELS, *NEURAL_BASELINES,
                 *EXTRA_NEURAL_BASELINES,
                 *EXTRA_ML_BASELINES, *OPTIONAL_ML_BASELINES):
        return "standard", "mask"
    raise ValueError(f"No predefined Stage04 preprocessing policy for {model}")

def prepare(panel: Panel, kind: str, missing_mode: str) -> dict:
    """Fixed neural preprocessing: raw zero + missing mask + Train Standard.

    Normalizers are fitted on observed Train values BEFORE any filling. NaNs
    then receive raw value 0; its transformed value is (0-center)/scale.
    The separate mask tells the model which zeros are missing, not measured.
    Never impute with the Train mean or run alternative policies in Stage04.
    """
    if (kind, missing_mode) != ("standard", "mask"):
        raise ValueError("Stage04 neural preprocessing is fixed to standard/mask")
    # Stage 03 already owns the three historical-target feature columns.
    # Never append panel.y (future labels) as new model input channels.
    raw = panel.x
    center, scale = fit_normalizer(raw[:panel.train_count], kind)
    observed = np.isfinite(raw)
    x = (np.where(observed, raw, 0.) - center) / scale
    missing = (~observed).astype(np.float32)
    yc, ys = panel.y[:panel.train_count].mean((0, 1)), panel.y[:panel.train_count].std((0, 1))
    ys = np.where(ys > 1e-8, ys, 1.)
    y = (panel.y - yc) / ys
    origins = np.arange(LOOKBACK, panel.train_count - HORIZON + 1)
    # Only three explicitly lagged columns of row t are used as the anchor.
    # In particular, profitability is the Stage03 past-only proxy, not the
    # future-annual-GOS-contaminated retrospective Target of quarter t-1.
    train_anchor = (past_scores(panel, origins) - yc) / ys
    dev_origins = np.arange(panel.train_count, len(panel.quarters))
    val_anchor = (past_scores(panel, dev_origins) - yc) / ys
    if not np.isfinite(train_anchor).all() or not np.isfinite(val_anchor).all():
        raise ValueError("Stage 03 previous-target input history is incomplete")
    out = {
        "train_x": np.stack([x[o - LOOKBACK:o] for o in origins]),
        "train_mask": np.stack([missing[o - LOOKBACK:o] for o in origins]),
        "train_y": np.stack([y[o:o + HORIZON] for o in origins]),
        "train_last": train_anchor,
        "train_context": known_score_context(panel, origins, yc, ys),
        "val_x": np.stack([x[o - LOOKBACK:o] for o in dev_origins]),
        "val_mask": np.stack([missing[o - LOOKBACK:o] for o in dev_origins]),
        "val_y": np.stack([y[o:o + HORIZON] for o in dev_origins]),
        "val_last": val_anchor,
        "val_context": known_score_context(panel, dev_origins, yc, ys),
        "x_center": center, "x_scale": scale, "y_center": yc, "y_scale": ys,
        "origins": origins, "val_origins": dev_origins,
    }
    return {k: v.astype(np.float32) if k not in ("origins", "val_origins")
            else v for k, v in out.items()}

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

    All three axes, all output modes and every ML/GNN family get exactly
    the same 9 known values. No current-quarter external features or Y[t].
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

def train_graph(panel: Panel, neighbours: int = 8) -> np.ndarray:
    """Fixed graph from Train-only inter-industry three-axis correlations.

    This is statistical association, not causal edges or a BEA IO graph.
    Absolute correlation is averaged over three axes; retain symmetric top-k
    neighbours, and self-loops. Learned graph architectures estimate their
    separate adjacency only through the training loss.
    """
    matrices = []
    # The same past-only score estimates used for forecasting define the
    # learned/fixed training-period graph. No retrospective Target is an input.
    history = past_scores(panel, np.arange(1, panel.train_count + 1))
    for axis in range(3):
        matrices.append(np.nan_to_num(np.corrcoef(history[:, :, axis].T), nan=0.))
    strength = np.mean(np.abs(matrices), axis=0)
    np.fill_diagonal(strength, 0.)
    selected = np.argsort(strength, axis=1)[:, -neighbours:]
    mask = np.zeros_like(strength, bool)
    np.put_along_axis(mask, selected, True, axis=1)
    a = np.where(mask | mask.T, strength, 0.)
    np.fill_diagonal(a, 1.)
    return a.astype(np.float32)

def baseline_prediction(panel: Panel, kind: str) -> np.ndarray:
    dev_origins = np.arange(panel.train_count, len(panel.quarters))
    if kind == "Persistence":
        return past_scores(panel, dev_origins)
    if kind == "SeasonalNaive":
        return origin_lag_scores(panel, dev_origins, 4)
    raise ValueError(kind)

def metrics(pred: np.ndarray, truth: np.ndarray, persistence: np.ndarray) -> dict:
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction shape/values")
    d = pred - truth
    ici = d.mean(-1)
    mae = float(np.abs(d).mean())
    pmae = float(np.abs(persistence - truth).mean())
    result = dict(axis_mae=mae, axis_rmse=float(np.sqrt(np.square(d).mean())),
                  ici_mae=float(np.abs(ici).mean()), ici_rmse=float(np.sqrt(np.square(ici).mean())),
                  skill_vs_persistence=1 - mae / pmae if pmae else 0.)
    for i, name in enumerate(("profitability", "productivity", "growth")):
        result[f"{name}_mae"] = float(np.abs(d[..., i]).mean())
        p, y = pred[..., i], truth[..., i]
        variance = float(y.std(axis=0).mean())
        result[f"{name}_time_std_ratio"] = float(p.std(axis=0).mean()) / variance if variance else 0.
        dp, dy = np.diff(p, axis=0).ravel(), np.diff(y, axis=0).ravel()
        result[f"{name}_qoq_corr"] = float(np.corrcoef(dp, dy)[0, 1]) if dp.std() > 1e-12 and dy.std() > 1e-12 else 0.
        result[f"{name}_direction"] = float((np.sign(dp) == np.sign(dy)).mean())
    return result

def model_registry() -> list[dict]:
    refs = {
        "MTGNN": "https://github.com/nnzhan/MTGNN",
        "B-MTGNN": "https://github.com/zaidalmahmoud/Cyber-trend-forecasting",
        "GraphWaveNet": "https://github.com/nnzhan/Graph-WaveNet",
        "AGCRN": "https://github.com/LeiBAI/AGCRN",
        "DCRNN": "https://github.com/liyaguang/DCRNN",
        "STGCN": "https://github.com/VeritasYin/STGCN_IJCAI-18",
        "T-GCN": "https://github.com/lehaifeng/T-GCN",
        "GConvGRU": "https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html",
        "GConvLSTM": "https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html",
        "EvolveGCN-H-GRU": "https://github.com/IBM/EvolveGCN",
        "EvolveGCN-O-GRU": "https://github.com/IBM/EvolveGCN",
    }
    return [dict(
        model=name, family="GNN" if name in GNN_MODELS else "baseline",
        implementation=("PRISM equation-level forecasting adaptation" if name in SPECIAL
                        else "spatial operator + temporal backbone adaptation" if name in GNN_MODELS
                        else "non-graph forecast baseline"),
        reference=refs.get(name, "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html"
                           if name in GNN_MODELS else ""),
        learned_graph=name in ("MTGNN", "B-MTGNN", "GraphWaveNet", "AGCRN"),
        posterior_samples=30 if name == "B-MTGNN" else 1,
    ) for name in (*GNN_MODELS, *BASELINES)]

def row_normalize(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1, keepdim=True).clamp_min(1e-8)

def symmetric_normalize(a: torch.Tensor) -> torch.Tensor:
    d = a.sum(-1).clamp_min(1e-8).rsqrt()
    return d[:, None] * a * d[None, :]

class SpatialOperator(nn.Module):
    """Distinct published spatial equations, dense adjacency A[target, source]."""
    def __init__(self, kind: str, width: int):
        super().__init__()
        self.kind, self.width = kind, width
        if kind not in ("ChebNet", "TAGCN", "GIN", "APPNP"):
            self.linear = nn.Linear(width, width)
        if kind in ("GraphSAGE", "GraphConv", "ARMA"):
            self.other = nn.Linear(width, width)
        if kind in ("ChebNet", "TAGCN"):
            self.poly = nn.Linear(3 * width, width)
        if kind in ("GIN", "APPNP", "ARMA"):
            self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        if kind in ("GAT", "GATv2", "GraphTransformer"):
            self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        if kind in ("GAT", "GATv2"):
            self.att = nn.Parameter(torch.empty(width))
            nn.init.normal_(self.att, std=width ** -0.5)
        if kind == "GIN":
            self.epsilon = nn.Parameter(torch.tensor(0.))
        if kind == "AGNN":
            self.beta = nn.Parameter(torch.tensor(1.))
        if kind == "GatedGraph":
            self.recur = nn.GRUCell(width, width)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(a.size(0), device=a.device, dtype=a.dtype)
        no_self = a * (1. - eye)
        a = no_self + eye
        r, s = row_normalize(a), symmetric_normalize(a)
        kind = self.kind
        if kind == "GCN":
            return self.linear(s @ x)
        if kind == "GraphSAGE":
            return self.linear(x) + self.other(row_normalize(no_self) @ x)
        if kind == "GraphConv":
            return self.linear(x) + self.other(no_self @ x)
        if kind == "GIN":
            return self.mlp((1 + self.epsilon) * x + no_self @ x)
        if kind in ("GAT", "GATv2", "GraphTransformer", "AGNN"):
            if kind != "AGNN":
                q, k, v = self.q(x), self.k(x), self.v(x)
            if kind == "GAT":
                scores = F.leaky_relu((q * self.att).sum(-1)[..., :, None]
                                     + (k * self.att).sum(-1)[..., None, :], .2)
            elif kind == "GATv2":
                scores = (F.leaky_relu(q[..., :, None, :] + k[..., None, :, :], .2)
                          * self.att).sum(-1)
            elif kind == "AGNN":
                z = F.normalize(x, dim=-1)
                scores = self.beta * (z @ z.transpose(-1, -2))
                v = x
            else:
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
            weights = scores.masked_fill(a == 0, -torch.inf).softmax(-1)
            return self.linear(weights @ v)
        if kind == "ChebNet":
            t1 = -(s @ x)  # scaled normalized Laplacian, lambda_max=2
            t2 = -2 * (s @ t1) - x
            return self.poly(torch.cat([x, t1, t2], -1))
        if kind == "SGC":
            return self.linear(s @ (s @ x))
        if kind == "TAGCN":
            h1 = s @ x
            return self.poly(torch.cat([x, h1, s @ h1], -1))
        if kind == "APPNP":
            h0 = self.mlp(x)
            h = h0
            for _ in range(5):
                h = .9 * (s @ h) + .1 * h0
            return h
        if kind == "GCNII":
            h = x
            for i in range(2):
                z = .9 * (s @ h) + .1 * x
                beta = math.log(.5 / (i + 1) + 1.)
                h = F.relu((1 - beta) * z + beta * self.linear(z))
            return h
        if kind == "ARMA":
            stack = []
            for skip in (self.other(x), self.mlp(x)):
                h = x
                for _ in range(3):
                    h = F.relu(self.linear(s @ h) + skip)
                stack.append(h)
            return (stack[0] + stack[1]) / 2
        if kind == "GatedGraph":
            h = x
            for _ in range(3):
                msg = self.linear(no_self @ h)
                h = self.recur(msg.reshape(-1, self.width), h.reshape(-1, self.width)).reshape_as(x)
            return h
        raise ValueError(kind)

class TemporalHead(nn.Module):
    def __init__(self, kind: str, width: int, horizon: int, lookback: int):
        super().__init__()
        self.kind, self.horizon = kind, horizon
        if kind == "GRU":
            self.core = nn.GRU(width, width, batch_first=True)
        elif kind == "LSTM":
            self.core = nn.LSTM(width, width, batch_first=True)
        elif kind == "TCN":
            self.core = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d) for d in (1, 2, 4)])
        elif kind == "DLinear":
            self.core = nn.Linear(lookback*width, horizon*3)
        elif kind == "NBEATS":
            self.core = nn.Sequential(
                nn.Linear(lookback*width, 2*width), nn.GELU(),
                nn.Linear(2*width, 2*width), nn.GELU(),
                nn.Linear(2*width, horizon*3))
        else:
            layer = nn.TransformerEncoderLayer(width, 2, 2 * width, dropout=.1,
                                               batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.position = nn.Parameter(torch.randn(1, lookback, width) * .02)
        self.out = nn.Linear(width, horizon * 3)

    def forward(self, x):
        b, t, n, h = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, h)
        if self.kind == "GRU":
            _, z = self.core(x)
            z = z[-1]
        elif self.kind == "LSTM":
            _, (h, _) = self.core(x)
            z = h[-1]
        elif self.kind == "TCN":
            z = x.transpose(1, 2)
            for layer in self.core:
                z = z + F.gelu(layer(F.pad(z, (2 * layer.dilation[0], 0))))
            z = z[:, :, -1]
        elif self.kind in ("DLinear", "NBEATS"):
            direct=self.core(x.reshape(b*n,t*h))
            return direct.reshape(b,n,self.horizon,3).permute(0,2,1,3)
        else:
            z = self.core(x + self.position)[:, -1]
        return self.out(z).reshape(b, n, self.horizon, 3).permute(0, 2, 1, 3)

class SpatialTemporal(nn.Module):
    def __init__(self, spatial, temporal, width, horizon, lookback):
        super().__init__()
        self.ops = nn.ModuleList([SpatialOperator(spatial, width) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.head = TemporalHead(temporal, width, horizon, lookback)

    def forward(self, x, a):
        for op, norm in zip(self.ops, self.norms):
            x = norm(x + F.dropout(F.gelu(op(x, a)), .1, self.training))
        return self.head(x)

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
    """Core adapted from Wu's MIT-licensed MTGNN (notice in prism04_LICENSE.txt)."""
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

class GraphWaveNet(nn.Module):
    """Dilated gated temporal stack + forward/reverse/adaptive diffusion."""
    def __init__(self, nodes, width, horizon):
        super().__init__()
        self.horizon = horizon
        self.e1, self.e2 = (nn.Parameter(torch.randn(nodes, 8) * .1) for _ in range(2))
        self.filters = nn.ModuleList([nn.Conv2d(width, width, (1, 2), dilation=(1, d)) for d in (1, 2, 4, 1)])
        self.gates = nn.ModuleList([nn.Conv2d(width, width, (1, 2), dilation=(1, d)) for d in (1, 2, 4, 1)])
        self.mix = nn.ModuleList([nn.Conv2d(7 * width, width, 1) for _ in range(4)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.out = nn.Linear(width, horizon * 3)
        self.last_graph = None

    def forward(self, x, a):
        adaptive = F.relu(self.e1 @ self.e2.T).softmax(-1)
        self.last_graph = adaptive.detach()
        supports = (row_normalize(a), row_normalize(a.T), adaptive)
        x, skip = x.permute(0, 3, 2, 1), 0
        for filt, gate, mix, norm in zip(self.filters, self.gates, self.mix, self.norm):
            padded = F.pad(x, (filt.dilation[1], 0))
            z = torch.tanh(filt(padded)) * torch.sigmoid(gate(padded))
            parts = [z]
            for support in supports:
                first = torch.einsum("bcnt,mn->bcmt", z, support)
                parts.extend([first, torch.einsum("bcnt,mn->bcmt", first, support)])
            x = norm((x + F.dropout(mix(torch.cat(parts, 1)), .1, self.training)).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            skip = skip + x[..., -1]
        z = self.out(F.relu(skip.transpose(1, 2)))
        return z.reshape(z.size(0), z.size(1), self.horizon, 3).permute(0, 2, 1, 3)

class AdaptiveConv(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=8):
        super().__init__()
        self.weight_pool = nn.Parameter(torch.empty(embed_dim, 2, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_pool)

    def forward(self, x, embedding, a):
        weights = torch.einsum("ne,ekio->nkio", embedding, self.weight_pool)
        xg = torch.stack([x, a @ x], dim=2)
        return torch.einsum("bnki,nkio->bno", xg, weights) + embedding @ self.bias_pool

class AGCRN(nn.Module):
    def __init__(self, nodes, width, horizon):
        super().__init__()
        self.width, self.horizon = width, horizon
        self.embedding = nn.Parameter(torch.randn(nodes, 8) * .1)
        self.gate = AdaptiveConv(2 * width, 2 * width)
        self.update = AdaptiveConv(2 * width, width)
        self.out = nn.Linear(width, horizon * 3)
        self.last_graph = None

    def forward(self, x, unused):
        a = F.relu(self.embedding @ self.embedding.T).softmax(-1)
        self.last_graph = a.detach()
        h = torch.zeros_like(x[:, 0])
        for t in range(x.size(1)):
            z, r = torch.sigmoid(self.gate(torch.cat([x[:, t], h], -1), self.embedding, a)).chunk(2, -1)
            candidate = torch.tanh(self.update(torch.cat([x[:, t], r * h], -1), self.embedding, a))
            h = z * h + (1 - z) * candidate
        return self.out(h).reshape(x.size(0), x.size(2), self.horizon, 3).permute(0, 2, 1, 3)

class DiffusionCell(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(10 * width, 2 * width)
        self.update = nn.Linear(10 * width, width)

    def diff(self, x, a):
        parts = [x]
        for b in (row_normalize(a), row_normalize(a.T)):
            h = b @ x
            parts.extend([h, b @ h])
        return torch.cat(parts, -1)

    def forward(self, x, h, a):
        z, r = torch.sigmoid(self.gate(self.diff(torch.cat([x, h], -1), a))).chunk(2, -1)
        c = torch.tanh(self.update(self.diff(torch.cat([x, r * h], -1), a)))
        return z * h + (1 - z) * c

class DCRNN(nn.Module):
    """Diffusion-convolution GRU encoder/decoder; no held-out teacher forcing."""
    def __init__(self, width, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder, self.decoder = DiffusionCell(width), DiffusionCell(width)
        self.feedback, self.out = nn.Linear(3, width), nn.Linear(width, 3)

    def forward(self, x, a):
        h = torch.zeros_like(x[:, 0])
        for t in range(x.size(1)):
            h = self.encoder(x[:, t], h, a)
        prev = torch.zeros((*h.shape[:2], 3), device=h.device)
        results = []
        for _ in range(self.horizon):
            h = self.decoder(self.feedback(prev), h, a)
            prev = self.out(h)
            results.append(prev)
        return torch.stack(results, 1)

class GraphRecurrent(nn.Module):
    """Graph operators INSIDE recurrent gates, not GCN followed by a plain GRU.

    T-GCN uses first-order normalized convolution; GConvGRU/LSTM use a
    third-order Chebyshev basis. LSTM includes learned diagonal peepholes.
    A common direct 12x3 head replaces the task-specific paper output head.
    """
    def __init__(self, kind, width, horizon):
        super().__init__()
        self.kind, self.width, self.horizon = kind, width, horizon
        basis = 1 if kind == "T-GCN" else 3
        self.gates = nn.Linear(2 * width * basis, (4 if kind == "GConvLSTM" else 2) * width)
        if kind != "GConvLSTM":
            self.update = nn.Linear(2 * width * basis, width)
        else:
            self.peepholes = nn.Parameter(torch.zeros(3, width))
        self.out = nn.Linear(width, horizon * 3)

    def basis(self, x, s):
        if self.kind == "T-GCN":
            return s @ x
        first = -(s @ x)
        return torch.cat([x, first, -2 * (s @ first) - x], -1)

    def forward(self, x, a):
        s = symmetric_normalize(a)
        h, c = torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, 0])
        for t in range(x.size(1)):
            joined = self.basis(torch.cat([x[:, t], h], -1), s)
            if self.kind == "GConvLSTM":
                i, f, o, g = self.gates(joined).chunk(4, -1)
                i = torch.sigmoid(i + c * self.peepholes[0])
                f = torch.sigmoid(f + c * self.peepholes[1])
                c = f * c + i * torch.tanh(g)
                h = torch.sigmoid(o + c * self.peepholes[2]) * torch.tanh(c)
            else:
                z, r = torch.sigmoid(self.gates(joined)).chunk(2, -1)
                candidate = torch.tanh(self.update(self.basis(torch.cat([x[:, t], r * h], -1), s)))
                h = z * h + (1 - z) * candidate
        return self.out(h).reshape(x.size(0), x.size(2), self.horizon, 3).permute(0, 2, 1, 3)

class EvolvingGraph(nn.Module):
    """Evolving GCN weights with a shared GRU forecasting head (named explicitly).

    H variant: top-k node summaries drive a row-wise matrix GRU. O variant:
    autonomous row-wise matrix LSTM evolves weights. These are PRISM adapters,
    not unmodified EvolveGCN node-classification implementations.
    """
    def __init__(self, kind, width, horizon, lookback):
        super().__init__()
        self.kind, self.width = kind, width
        self.weight = nn.Parameter(torch.eye(width))
        if kind == "H":
            self.score = nn.Parameter(torch.randn(width) * .1)
            self.cell = nn.GRUCell(width, width)
        else:
            self.cell = nn.LSTMCell(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = TemporalHead("GRU", width, horizon, lookback)

    def forward(self, x, a):
        b, t, n, h = x.shape
        w = self.weight[None].expand(b, -1, -1)
        c = torch.zeros_like(w).reshape(b * h, h)
        s = symmetric_normalize(a)
        outputs = []
        for k in range(t):
            if self.kind == "H":
                scores = (x[:, k] @ self.score) / self.score.norm().clamp_min(1e-6)
                ids = scores.topk(h, dim=1).indices
                summary = x[:, k].gather(1, ids[..., None].expand(-1, -1, h))
                summary = summary * scores.gather(1, ids).tanh()[..., None]
                w = self.cell(summary.transpose(1, 2).reshape(b * h, h), w.reshape(b * h, h)).reshape(b, h, h)
            else:
                z, c = self.cell(w.reshape(b * h, h), (w.reshape(b * h, h), c))
                w = z.reshape(b, h, h)
            outputs.append(self.norm(F.leaky_relu((s @ x[:, k]) @ w, .1)))
        return self.head(torch.stack(outputs, 1))

class STGCN(nn.Module):
    def __init__(self, width, horizon):
        super().__init__()
        self.temporal = nn.ModuleList([nn.Conv2d(width, 2 * width, (1, 3)) for _ in range(2)])
        self.spatial = SpatialOperator("ChebNet", width)
        self.norm = nn.LayerNorm(width)
        self.out, self.horizon = nn.Linear(width, horizon * 3), horizon

    def forward(self, x, a):
        for i, temporal in enumerate(self.temporal):
            z = temporal(F.pad(x.permute(0, 3, 2, 1), (2, 0)))
            value, gate = z.chunk(2, dim=1)
            x = (value * torch.sigmoid(gate)).permute(0, 3, 2, 1)
            if i == 0:
                x = F.relu(self.spatial(x, a))
        z = self.out(self.norm(x[:, -1]))
        return z.reshape(z.size(0), z.size(1), self.horizon, 3).permute(0, 2, 1, 3)

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
        self.name = name
        self.values = nn.Linear(features, width)
        self.missing = nn.Linear(features, width, bias=False)
        self.norm = nn.LayerNorm(width)
        if name in ("MTGNN", "B-MTGNN"):
            self.core = MTGNNCore(nodes, width, lookback, horizon)
        elif name == "GraphWaveNet":
            self.core = GraphWaveNet(nodes, width, horizon)
        elif name == "AGCRN":
            self.core = AGCRN(nodes, width, horizon)
        elif name == "DCRNN":
            self.core = DCRNN(width, horizon)
        elif name == "STGCN":
            self.core = STGCN(width, horizon)
        elif name in ("T-GCN", "GConvGRU", "GConvLSTM"):
            self.core = GraphRecurrent(name, width, horizon)
        elif name.startswith("EvolveGCN-"):
            self.core = EvolvingGraph(name.split("-")[1], width, horizon, lookback)
        elif name in (*NEURAL_BASELINES, *EXTRA_NEURAL_BASELINES):
            self.core = TemporalHead(name, width, horizon, lookback)
        else:
            spatial, temporal = name.rsplit("-", 1)
            self.core = SpatialTemporal(spatial, temporal, width, horizon, lookback)
    def forward(self, x, missing, adjacency, last_target):
        z = self.norm(self.values(x) + self.missing(missing))
        delta = self.core(z) if self.name in (*NEURAL_BASELINES, *EXTRA_NEURAL_BASELINES) else self.core(z, adjacency)
        # The anchor is calculated only from Stage 03's lagged target features
        # in prepare(), not from the current-quarter target labels.
        return delta + last_target[:, None]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ensemble_selection_path(model_predictions: dict[str, np.ndarray],
                            truth: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Select a pool on the EXACT five-seed-mean forecasts used in the plots.

    Each model's deployable reference prediction is the MEAN of its five
    matched seeds. Each ensemble is the equal-weight mean of these model
    predictions. Hence both the selection MAE and the visualized forecast
    score exactly the SAME array, with no Jensen/aggregation discrepancy.

    The five separately scored seed-matched ensembles' MAE is retained as
    a robustness diagnostic, NOT silently substituted as the objective.
    Correlation-aware near-tie selection and the smallest within one paired
    industry-block SE of the minimum are descriptive size heuristics, not a
    time-series generalization confidence interval.
    """
    names = sorted(model_predictions)
    if not names:
        raise ValueError("Ensemble selection needs at least one model")
    y = np.asarray(truth, dtype=np.float64)
    if y.ndim != 3 or y.shape[-1] != 3 or not np.isfinite(y).all():
        raise ValueError("Expected finite Validation truth [H,N,3]")
    shape = (5,) + y.shape
    predictions = {}
    for name, data in model_predictions.items():
        p = np.asarray(data, dtype=np.float64)
        if p.shape != shape or not np.isfinite(p).all():
            raise ValueError(f"{name}: invalid five-seed predictions {shape}")
        predictions[name] = p
    # Both plotting and model selection use these exact deployed predictions.
    averaged = {name: predictions[name].mean(axis=0) for name in names}

    errors = np.stack([(averaged[name] - y).reshape(-1) for name in names])
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(errors) if len(names) > 1 else np.ones((1, 1))
    matrix = np.nan_to_num(matrix, nan=1., posinf=1., neginf=-1.)
    matrix = np.clip((matrix + matrix.T) / 2., -1., 1.)
    np.fill_diagonal(matrix, 1.)
    corr = pd.DataFrame(matrix, index=names, columns=names)

    def node_losses(p):
    # Input shape [12 sequential one-quarter Dev origins, 59 industries, 3 axes].
        return np.abs(p - y).mean(axis=(0, 2))

    def paired_se(a, b):
        difference = a - b
        return float(np.std(difference, ddof=1) / np.sqrt(len(difference)))

    path, audit, selected = [], [], []
    current_sum = np.zeros(y.shape, dtype=np.float64)
    current_seed_sum = np.zeros(shape, dtype=np.float64)
    for k in range(1, len(names) + 1):
        trials = []
        for name in names:
            if name in selected:
                continue
            p = (current_sum + averaged[name]) / k
            losses = node_losses(p)
            score = float(losses.mean())
            separately_scored = (
                np.abs((current_seed_sum + predictions[name]) / k - y[None]).mean())
            if selected:
                a = (current_sum / len(selected) - y).reshape(-1)
                b = (averaged[name] - y).reshape(-1)
                correlation = (float(np.corrcoef(a, b)[0, 1])
                               if min(np.std(a), np.std(b)) > 1e-12 else 1.)
                correlation = correlation if np.isfinite(correlation) else 1.
            else:
                correlation = 0.
            trials.append(dict(name=name, score=score, losses=losses,
                               score_separate_seeds=float(separately_scored),
                               correlation=correlation))
        best = min(trials, key=lambda t: (t["score"], t["name"]))
        for trial in trials:
            trial["se"] = paired_se(trial["losses"], best["losses"])
            trial["eligible"] = trial["score"] - best["score"] <= trial["se"] + 1e-12
        near = [t for t in trials if t["eligible"]]
        pick = min(near, key=lambda t: (t["correlation"], t["score"], t["name"]))
        for trial in trials:
            audit.append(dict(
                k=k, candidate_model=trial["name"],
                trial_axis_mae=trial["score"],
                trial_separately_scored_seed_mae=trial["score_separate_seeds"],
                corr_with_current_pool=trial["correlation"],
                best_trial_axis_mae=best["score"],
                paired_se_from_best=trial["se"],
                within_1se_of_best=trial["eligible"],
                chosen=trial["name"] == pick["name"],
            ))
        selected.append(pick["name"])
        current_sum += averaged[pick["name"]]
        current_seed_sum += predictions[pick["name"]]
        ensemble = current_sum / k
        node_mae = node_losses(ensemble)
        seedwise_ensemble = current_seed_sum / k
        by_seed = np.abs(seedwise_ensemble - y[None]).mean(axis=(1, 2, 3))
        pair_corr = [corr.loc[a, b] for j, a in enumerate(selected)
                     for b in selected[j + 1:]]
        path.append(dict(
            k=k, added_model=pick["name"], models="|".join(selected),
            axis_mae=float(node_mae.mean()),
            axis_mae_separately_scored_seed_mean=float(by_seed.mean()),
            axis_mae_seed_sd=float(by_seed.std(ddof=1)),
            ici_mae=float(np.abs((ensemble - y).mean(-1)).mean()),
            mean_pairwise_error_corr=float(np.mean(pair_corr)) if pair_corr else np.nan,
            node_mae=node_mae,
        ))

    best_index = min(range(len(path)), key=lambda i: (path[i]["axis_mae"], i))
    best_nodes = path[best_index]["node_mae"]
    min_mae = path[best_index]["axis_mae"]
    for entry in path:
        entry["mae_gap_from_path_min"] = entry["axis_mae"] - min_mae
        entry["paired_industry_one_se"] = paired_se(entry["node_mae"], best_nodes)
        entry["within_one_se_of_min"] = (
            entry["mae_gap_from_path_min"] <= entry["paired_industry_one_se"] + 1e-12)
        del entry["node_mae"]
    k = next(row["k"] for row in path if row["within_one_se_of_min"])
    for row in path:
        row["selected_k"] = k
        row["global_min_k"] = best_index + 1
        row["global_min_mae"] = min_mae
    return pd.DataFrame(path), pd.DataFrame(audit), corr, selected[:k]

def economic_components(panel: Panel) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct next-quarter baselines from Stage01's historical components.

    Profitability MUST use per-origin Stage03 history. Once a past year ends
    its annual GOS becomes usable, so t-1 and t-2 must be reconstructed
    using the same forecast origin. Never use retrospective Target labels.
    The latest revised official source values remain the fixed data policy.
    Only the already completed 2005 annual GOS is read to build the
    2006Q1-2008Q4 calibration history. No future profitability label or
    Stage03 Test is read here.
    """
    periods = pd.period_range("2005Q1", "2022Q4", freq="Q").astype(str)
    months = pd.period_range("2005-01", "2022-12", freq="M").astype(str)
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
    employment = monthly.to_numpy(dtype=np.float64).reshape(72,3,NODE_COUNT).mean(1)
    combined = (source.loc[source.series.eq("employment_523_525_aggregate")]
                .set_index("period").value.reindex(months))
    if combined.isna().any() or (combined<=0).any():
        raise RuntimeError("Incomplete Stage01 funds employment aggregate")
    funds, securities = (panel.nodes.index("funds_trusts"),
                         panel.nodes.index("securities"))
    employment[:,funds] = combined.to_numpy(dtype=np.float64).reshape(72,3).mean(1)
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
        input_source_period=["2005Q1", "2022Q4"],
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
    Alpha is fixed using all 2006-2019 Train labels, never Dev/Test.
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
    """Freeze alpha from ALL 2006-2019 Train years, never from Dev/Test.

    An older implementation withheld the final two Train years AND started
    at 2009Q1 due to the neural 12-quarter lookback. Neither exclusion is
    needed for a two-quarter economic prior, so all 56 Train quarters now
    participate equally in the 14 year-block paired 1-SE comparison.
    A partial fit_end is explicitly rejected to prevent silent reversion.
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
        if self.architecture in (*NEURAL_BASELINES, *EXTRA_NEURAL_BASELINES):
            correction = self.base.core(z)
        else:
            correction = self.base.core(z, graph)
        # The labels are ALWAYS the exact same Stage03 Y[t]. Only the
        # network-output reference differs. Direct is NOT a delta forecast.
        if self.prediction_mode == "direct":
            anchor = torch.zeros_like(last)
        elif self.prediction_mode == "residual":
            anchor = last
        else:
            anchor = prior
        return anchor[:,None]+correction+self.change_head(context)[:,None]

def prepare_three_axis_fold(panel: Panel, structure: np.ndarray,
                            fit_end: int, val_first: int,
                            val_end: int, scaler: str = "standard",
                            missing_mode: str = "mask") -> dict:
    if (structure.shape != panel.y.shape
            or not 20 <= fit_end <= TRAIN_COUNT
            or not fit_end <= val_first < val_end <= TOTAL_COUNT):
        raise ValueError("Invalid Train-only three-axis structure fold")
    if (scaler, missing_mode) not in (
            ("standard", "mask"), ("none", "native"), ("none", "mask")):
        raise ValueError("Only predefined Stage04 family policies are permitted")
    if scaler == "standard":
        center, width = fit_normalizer(panel.x[:fit_end], "standard")
    else:
        center = np.zeros(panel.x.shape[-1], dtype=float)
        width = np.ones(panel.x.shape[-1], dtype=float)
    observed = np.isfinite(panel.x)
    # Impute original NaNs with RAW 0 before Standard normalization.
    # Missingness is retained independently. In native mode the zero-filled
    # history serves only as an intermediate tensor; _ml_design_matrix
    # restores the genuine NaNs, without adding mask channels for trees.
    x = ((np.where(observed,panel.x,0.)-center)/width).astype(np.float32)
    nan_mask=(~observed).astype(np.float32)
    masks=nan_mask if missing_mode=="mask" else np.zeros_like(x,dtype=np.float32)
    yc = panel.y[:fit_end].mean((0,1))
    ys = np.where(panel.y[:fit_end].std((0,1)) > 1e-8,
                  panel.y[:fit_end].std((0,1)),1.)
    y = ((panel.y-yc)/ys).astype(np.float32)
    structural = ((structure-yc)/ys).astype(np.float32)
    train = np.arange(LOOKBACK,fit_end)
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

def three_axis_weighted_huber(pred, truth, last, ys):
    error = (pred-truth)*ys
    changed = (truth-last[:,None])*ys
    weights = 1.+torch.minimum(
        changed.abs()/(ys+1e-8),torch.full_like(changed,2.))
    weights = weights/weights.mean().detach()
    cost = F.smooth_l1_loss(
        error, torch.zeros_like(error),beta=3.,reduction="none")
    return (cost*weights).mean()

def mtgnn_bayesian_inference(model, z, graph, name, seed, yc, ys):
    """Return deployable mean + across-dropout predictive uncertainty.

    Faithful distinction from Almahmoud et al.'s B-MTGNN implementation:
    B-MTGNN leaves dropout ACTIVE for 30 MC draws, MTGNN explicitly turns
    dropout OFF and takes ONE deterministic evaluation. The underlying
    MTGNNCore and training loss are unchanged for a controlled comparison.
    The rest of Stage04's GNNs retain their previous 30 stochastic draws:
    their old prediction/checkpoint files are not invalidated by this fix.
    All random draws are performed on the same historical Validation input,
    with no label access or refit during inference.
    """
    if name == "B-MTGNN":
        model.train()
        n_draws=STAGE04_BMTGNN_MC_SAMPLES
        regime="B_MTGNN_MC_DROPOUT"
    elif name == "MTGNN":
        model.eval()
        n_draws=STAGE04_MTGNN_DETERMINISTIC_SAMPLES
        regime="MTGNN_DETERMINISTIC"
    else:
        model.train()
        n_draws=30
        regime="OTHER_GNN_LEGACY_MC_UNCHANGED"
    seed_all(seed+100000)
    with torch.inference_mode():
        sampled=np.stack([
            model(z["val_x"],z["val_mask"],graph,
                  z["val_last"],z["val_structure"],z["val_context"])
            .detach().cpu().numpy()[:,0] for _ in range(n_draws)])
    means=(sampled.mean(axis=0)*ys+yc).astype(np.float64)
    if (means.shape!=(z["val_x"].shape[0],NODE_COUNT,3)
            or not np.isfinite(means).all()):
        raise RuntimeError("Invalid MTGNN/B-MTGNN posterior-mean prediction")
    details=dict(
        prediction=means,
        inference_regime=regime,
        inference_draws=int(n_draws))
    if name == "B-MTGNN":
        if n_draws < 2:
            raise RuntimeError("Bayesian inference needs multiple draws")
        stdev=(sampled.std(axis=0,ddof=1)*ys).astype(np.float64)
        if (not np.isfinite(stdev).all()
                or not np.any(stdev > 1e-8)):
            raise RuntimeError(
                "B-MTGNN dropout draws are identical: Bayes inference is OFF")
        details.update(predictive_mc_std=stdev,
                       predictive_mc_var=stdev**2)
    return details

def fit_three_axis_gnn(panel, data, model_name, seed, device,
                       fixed_epochs: int | None,
                       prediction_mode: str = "structural"):
    """One identical algorithm for 41 GNNs and 3 temporal neural baselines."""
    seed_all(seed)
    z = {k:torch.as_tensor(data[k],device=device)
        for k in ("train_x","train_mask","train_y","train_last",
                   "train_context","train_structure","val_x","val_mask","val_y",
                   "val_last","val_context","val_structure")}
    yc,ys = data["y_center"],data["y_scale"]
    y_scale = torch.as_tensor(ys,device=device)
    cutoff = int(data["train_origins"][-1])+1
    gpanel = Panel(panel.x,panel.y,panel.quarters,panel.nodes,
                   panel.names,panel.features,cutoff,panel.hashes)
    graph = torch.as_tensor(train_graph(gpanel),device=device)
    model = ThreeAxisStructureGNN(
        model_name,data["x_center"],data["x_scale"],yc,ys,
        num_features=data["train_x"].shape[-1],
        nodes=len(panel.nodes),
        prediction_mode=prediction_mode).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),lr=FIXED_LEARNING_RATE,weight_decay=1e-4)
    iterations = (STAGE04_ACTIVE_EPOCHS if fixed_epochs is None
                  else int(fixed_epochs))
    best,epoch_of_best,stale = float("inf"),0,0
    start = time.perf_counter()
    for epoch in range(1,iterations+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        pred = model(z["train_x"],z["train_mask"],graph,
                     z["train_last"],z["train_structure"],z["train_context"])
        loss = three_axis_weighted_huber(
            pred,z["train_y"],z["train_last"],y_scale)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite common GNN training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.,error_if_nonfinite=True)
        optim.step()
        if fixed_epochs is None:
            # The Bayesian variant selects epochs using an MC-dropout
            # posterior mean; MTGNN uses deterministic eval-mode validation.
            model.train() if model_name == "B-MTGNN" else model.eval()
            with torch.inference_mode():
                draws=(STAGE04_BMTGNN_INNER_MC_SAMPLES
                       if model_name == "B-MTGNN" else 1)
                future=torch.stack([
                    model(z["val_x"],z["val_mask"],graph,
                          z["val_last"],z["val_structure"],z["val_context"])
                    for _ in range(draws)]).mean(dim=0)
                value=float(((future-z["val_y"]).abs()*y_scale).mean())
            if value < best-1e-6:
                best,epoch_of_best,stale=value,epoch,0
            else:
                stale+=1
            if stale>=STAGE04_ACTIVE_PATIENCE:
                break
    if fixed_epochs is None:
        del model,optim,z
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return dict(best_epoch=epoch_of_best,
                    train_only_validation_mae=best,
                    epochs_tried=epoch)
    evaluated=mtgnn_bayesian_inference(
        model,z,graph,model_name,seed,yc,ys)
    prediction=evaluated["prediction"]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite Stage04 GNN structural forecast")
    saved_state={k:v.detach().cpu().clone()
                 for k,v in model.state_dict().items()}
    total_seconds=time.perf_counter()-start
    nparams=sum(p.numel() for p in model.parameters())
    del model,optim,z
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return dict(**evaluated,state_dict=saved_state,
                parameters=nparams,seconds=total_seconds,
                best_epoch=fixed_epochs)

def full_baseline_registry() -> dict:
    """All 41 GNNs; six temporal baselines; three representative ML baselines.

    ML coverage: Ridge (linear), LightGBM (boosting), ExtraTrees (bagging).
    Never add optional expensive estimators depending on installed packages.
    """
    return dict(
        gnn=list(GNN_MODELS),
        original_temporal=list(NEURAL_BASELINES),
        extra_temporal=list(EXTRA_NEURAL_BASELINES),
        original_ml=["LightGBM"],
        extra_ml=list(EXTRA_ML_BASELINES),
        optional_ml_installed=[],
        optional_ml_unavailable=[],
        original_naive=["Persistence","SeasonalNaive"],
        extra_naive=list(EXTRA_NAIVE_BASELINES))

def full_job_grid(mode: str,registry: dict,
                  seeds: tuple[int,...] | list[int] = tuple(DEFAULT_SEEDS)):
    """Every model gets ONE immutable family-based preprocessing policy.

    Models/seeds/forecast modes are retained, but the obsolete 2-scaler ×
    2-missingness search is removed. Stage06 can test those choices later.
    """
    if mode not in STAGE04_OUTPUT_MODES:
        raise ValueError(mode)
    if list(seeds)!=DEFAULT_SEEDS:
        raise ValueError("FULL benchmark must use five matched original seeds")
    jobs=[]
    for model in (registry["gnn"]+registry["original_temporal"]+
                  registry["extra_temporal"]+registry["original_ml"]+
                  registry["extra_ml"]+registry["optional_ml_installed"]):
        scaler, missing = fixed_preprocessing_policy(model)
        for seed in seeds:
            jobs.append((mode,model,scaler,missing,seed))
    for model in (registry["original_naive"]+registry["extra_naive"]):
        scaler, missing = fixed_preprocessing_policy(model)
        jobs.append((mode,model,scaler,missing,seeds[0]))
    if len(jobs)!=len(set(jobs)):
        raise RuntimeError("Duplicate per-mode training-policy job")
    core=sum(
        (model in registry["gnn"]+registry["original_temporal"]+
         ["LightGBM"]+registry["original_naive"])
        for _,model,*_ in jobs)
    expected_core = (
        len(registry["gnn"]) + len(registry["original_temporal"])
        + len(registry["original_ml"])) * len(seeds) + len(registry["original_naive"])
    if core!=expected_core:
        raise RuntimeError(f"Original model/seed coverage changed: {core}/{expected_core}")
    expected = ((len(registry["gnn"]) + len(registry["original_temporal"])
                 + len(registry["extra_temporal"]) + len(registry["original_ml"])
                 + len(registry["extra_ml"]) + len(registry["optional_ml_installed"]))
                * len(seeds) + len(registry["original_naive"])
                + len(registry["extra_naive"]))
    if len(jobs) != expected or {model for _,model,*_ in jobs} != set(
            registry["gnn"]+registry["original_temporal"]+registry["extra_temporal"]+
            registry["original_ml"]+registry["extra_ml"]+
            registry["optional_ml_installed"]+registry["original_naive"]+
            registry["extra_naive"]):
        raise RuntimeError("Fixed preprocessing dropped a requested model or seed")
    if any((scaler, missing) != fixed_preprocessing_policy(model)
           for _, model, scaler, missing, _ in jobs):
        raise RuntimeError("Stage04 preprocessing was changed by the job grid")
    return jobs

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

def _ml_regressor(model,seed):
    """Three predeclared and representative classical ML baselines only."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    if model=="Ridge":
        return Ridge(alpha=10.,solver="lsqr")
    if model=="ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=60,max_depth=12,max_features=.5,
            n_jobs=4,random_state=seed,min_samples_leaf=3)
    if model=="LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            **FIXED_BOOSTING,min_child_samples=40,reg_lambda=1.,
            n_jobs=4,force_col_wise=True,verbosity=-1,
            random_state=seed,use_missing=True,zero_as_missing=False)
    raise ValueError(f"No baseline factory registered for {model}")

def _ml_design_matrix(panel,data,split,model,missing_mode,forecast_mode):
    """Same forecast origins/contexts, model-family fixed input treatment.

    Direct/residual do NOT have access to the additional economic prior.
    Standard/mask families get RAW-zero input plus binary mask columns.
    Native-NaN tree families get unscaled original numeric data and NaNs,
    with NO additional indicator channels or mean imputation.
    No other classical estimator is registered in Stage04.
    """
    if (data["scaler_policy"],missing_mode) != fixed_preprocessing_policy(model):
        raise ValueError(f"Noncanonical Stage04 preprocessing for {model}")
    raw=data[f"{split}_x"]                 # normalized X, [origins,12,59,157]
    B,L,N,D=raw.shape
    history=raw.transpose(0,2,1,3).reshape(B*N,L*D)
    context=data[f"{split}_context"].astype(np.float64)
    if data["scaler_policy"] == "none":
        # Trees use original scores, not even a standardized score context.
        # The 3 groups are: last-score, last-minus-lag2, last-minus-lag5.
        context=context*np.tile(data["y_scale"],3)[None,None,:]
        context[:,:,:3]+=data["y_center"]
    features=[history,context.reshape(B*N,9)]
    if missing_mode == "mask":
        indicator=data[f"{split}_raw_nan_mask"].transpose(0,2,1,3)
        features.append(indicator.reshape(B*N,L*D))
    if forecast_mode=="structural":
        structural=data[f"{split}_structure"].astype(np.float64)
        if data["scaler_policy"] == "none":
            structural=structural*data["y_scale"]+data["y_center"]
        features.append(structural.reshape(B*N,3))
    features.append(np.tile(np.eye(N,dtype=np.float32),(B,1)))
    x=np.concatenate(features,axis=-1).astype(np.float32)
    if missing_mode=="native":
        # Restore exact raw NaNs for all native-capable tree baselines.
        # The intermediate fold filled 0 only to support one reusable tensor;
        # the native tree's actual fit data never sees those placeholders.
        indicator=data[f"{split}_raw_nan_mask"].transpose(0,2,1,3)
        hidden=indicator.reshape(B*N,L*D)>0.
        x[:,:L*D][hidden]=np.nan
    if np.isinf(x).any() or (missing_mode!="native" and not np.isfinite(x).all()):
        raise RuntimeError("Invalid ML design tensor")
    return x

def fit_full_ml(panel,data,model,seed,mode,missing_policy):
    """3 equal-axis regressors, fixed recipe; return original-score forecast."""
    if mode not in STAGE04_OUTPUT_MODES:
        raise ValueError(mode)
    if model not in ("LightGBM",*EXTRA_ML_BASELINES,
                     *OPTIONAL_ML_BASELINES):
        raise ValueError(model)
    if (data["scaler_policy"],missing_policy) != fixed_preprocessing_policy(model):
        raise ValueError(f"Stage04 {model} must use its fixed family preprocessing")
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    X=_ml_design_matrix(panel,data,"train",model,missing_policy,mode)
    Xv=_ml_design_matrix(panel,data,"val",model,missing_policy,mode)
    ys=data["y_scale"]
    yc=data["y_center"]
    y=data["train_y"][:,0]
    def base(split):
        if mode=="direct":
            return np.zeros_like(data[f"{split}_last"])
        if mode=="residual":
            return data[f"{split}_last"]
        return data[f"{split}_structure"]
    ydiff=((y-base("train"))*ys).reshape(-1,3)
    models=[]
    pred=[]
    tic=time.perf_counter()
    for axis in range(3):
        reg=_ml_regressor(model,seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning)
            reg.fit(X,ydiff[:,axis])
        v=np.asarray(reg.predict(Xv),dtype=np.float64)
        if not np.isfinite(v).all():
            raise RuntimeError(f"{model} nonfinite predicted {axis}")
        models.append(reg)
        pred.append(v)
    level=(base("val")*ys+yc).reshape(-1,3)
    forecast=(level+np.column_stack(pred)).reshape(
        len(data["val_origins"]),NODE_COUNT,3)
    if not np.isfinite(forecast).all():
        raise RuntimeError("Nonfinite ML forecast")
    return dict(prediction=forecast,models=models,
                fit_seconds=time.perf_counter()-tic)

def three_axis_evaluation(pred,panel):
    target=panel.y[TRAIN_COUNT:TOTAL_COUNT]
    past=past_scores(panel,np.arange(TRAIN_COUNT,TOTAL_COUNT))
    difference=pred-target
    move=pred-past
    actual=target-past
    results=dict(
        axis_mae=float(np.abs(difference).mean()),
        ici_mae=float(np.abs(difference.mean(-1)).mean()))
    for axis,name in enumerate(("profitability","productivity","growth")):
        a,b=move[...,axis],actual[...,axis]
        correlation=(float(np.corrcoef(a.ravel(),b.ravel())[0,1])
                     if np.std(a)>1e-10 else 0.)
        results.update({
            f"{name}_mae":float(np.abs(difference[...,axis]).mean()),
            f"{name}_change_amplitude_ratio":float(
                np.abs(a).mean()/np.abs(b).mean()),
            f"{name}_change_correlation":correlation,
            f"{name}_direction_accuracy":float(
                (np.sign(a)==np.sign(b)).mean())})
    return results

def full_grid_paths():
    out=RESULTS
    return dict(
        root=out,metrics=out/"metrics",predictions=out/"predictions",
        checkpoints=out/"checkpoints",plots=out/"plots",
        manifest=out/"run_manifest.json",log=out/"execution.log")

def _full_job_id(job):
    mode,name,scaler,missing,seed=job
    return f"{mode}__{name}__{scaler}__{missing}__s{seed}"

def _full_original_job(job,registry):
    _,model,*_=job
    return model in (
        registry["gnn"]+registry["original_temporal"]+
        registry["original_ml"]+registry["original_naive"])

def _atomic_json(path,record):
    tmp=path.with_suffix(path.suffix+".writing")
    tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False,
                              allow_nan=False,default=str),encoding="utf-8")
    tmp.replace(path)

def _atomic_npz(path,**values):
    tmp=path.with_suffix(".writing.npz")
    np.savez_compressed(tmp,**values)
    tmp.replace(path)

def full_grid_setup(args):
    """Validate every score definition, output mode, source and run count."""
    reg=full_baseline_registry()
    if reg["optional_ml_unavailable"]:
        print("UNAVAILABLE_OPTIONAL_BASELINES",
              reg["optional_ml_unavailable"],flush=True)
    per_mode={mode:full_job_grid(mode,reg)
              for mode in STAGE04_OUTPUT_MODES}
    expected=sum(len(jobs) for jobs in per_mode.values())
    base_core_count=sum(
        _full_original_job(job,reg) for job in per_mode["direct"])
    expected_core=(len(reg["gnn"])+len(reg["original_temporal"])+
                   len(reg["original_ml"]))*len(DEFAULT_SEEDS)+len(reg["original_naive"])
    if (base_core_count != expected_core or
            len({len(jobs) for jobs in per_mode.values()}) != 1):
        raise RuntimeError("Fixed-policy core model/seed coverage is incomplete")
    policies={model:dict(scaler=fixed_preprocessing_policy(model)[0],
                         missing=fixed_preprocessing_policy(model)[1])
              for model in sorted({job[1] for jobs in per_mode.values() for job in jobs})}
    paths=full_grid_paths()
    if args.dry_run:
        print("STAGE04 THREE-MODE FIXED-PREPROCESSING GRID",flush=True)
        print("CORE PER MODE",base_core_count,"TOTAL CORE",3*base_core_count)
        print("ADDITIONAL PER MODE",len(per_mode["direct"])-base_core_count)
        print("FULL PER MODE",len(per_mode["direct"]),
              "FULL TOTAL",expected,flush=True)
        for key,v in reg.items():
            print(key,":",v,flush=True)
        print("FIXED PREPROCESSING:",STAGE04_PREPROCESSING_POLICY,flush=True)
        print("POLICIES BY MODEL:",policies,flush=True)
        print("NO TRAIN MEAN IMPUTATION; RAW ZERO + MISSING MASK "
              "FOR NON-TREES; NATIVE NaN FOR CAPABLE TREES",flush=True)
        print("FIVE MATCHED SEEDS; Direct/Residual/Structural; "
              "same three Targets; NO Dev preprocessing selection",flush=True)
        return None

    panel=load_panel(ROOT)
    n,d,identity=economic_components(panel)
    coef,train_grid=select_train_structural_priors(panel,n,d,identity)
    # There is ONE alpha fit over ALL Train, not a second partial-Train fit.
    # The separate inner fold below still chooses neural epochs without Dev.
    inner_coef=coef.copy()
    if not np.array_equal(coef,inner_coef):
        raise RuntimeError("Stage04 must use one frozen alpha in every Train fold")
    structured=np.zeros_like(panel.y)
    structured[12:]=component_structural_prior(
        n,d,identity,np.arange(12,68),coef)
    # No alpha adaptation across folds: inner and final priors are identical.
    inner_structured=structured.copy()
    signature=dict(
        stage="04_3MODES_FIXED_FAMILY_PREPROCESSING",
        core_per_head=base_core_count,
        total_core=base_core_count*len(STAGE04_OUTPUT_MODES),
        additive_baselines_per_head=len(per_mode["direct"])-base_core_count,
        all_per_head=len(per_mode["direct"]),
        total_runs=expected,
        matched_seeds=DEFAULT_SEEDS,
        output_modes=list(STAGE04_OUTPUT_MODES),
        preprocessing_policy=STAGE04_PREPROCESSING_POLICY,
        preprocessing_by_model=policies,
        preprocessing_dev_tuned=False,
        pre_normalization_missing_replacement="RAW_ZERO_WITH_SEPARATE_MASK",
        bmtgnn_inference="MC_DROPOUT_TRAIN_MODE",
        bmtgnn_prediction_draws=STAGE04_BMTGNN_MC_SAMPLES,
        bmtgnn_train_epoch_selection_draws=STAGE04_BMTGNN_INNER_MC_SAMPLES,
        mtgnn_inference="DETERMINISTIC_EVAL_MODE",
        mtgnn_prediction_draws=STAGE04_MTGNN_DETERMINISTIC_SAMPLES,
        other_gnn_inference="UNCHANGED_30_TRAIN_MODE_DRAWS",
        native_nan_supported_trees=list(NATIVE_NAN_TREES),
        model_registry=reg,
        source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        train_sha256=panel.hashes[
            "Data/03_integrated_dataset/raw_train.csv"],
        dev_sha256=panel.hashes[
            "Data/03_integrated_dataset/raw_val.csv"],
        economics_source_sha256=hashlib.sha256(
            STAGE04_COMPONENT_SOURCE.read_bytes()).hexdigest(),
        economics_score_bounds_sha256=hashlib.sha256(
            STAGE04_TARGET_BOUNDS.read_bytes()).hexdigest(),
        economic_prior_alpha_policy=STAGE04_ALPHA_POLICY,
        economic_prior_alpha_calibration_start="2006Q1",
        economic_prior_alpha_calibration_end="2019Q4",
        economic_prior_alpha_calibration_quarters=TRAIN_COUNT,
        economic_prior_alpha_is_adaptive=False,
        economic_prior_train_coefficient=coef.tolist(),
        economic_prior_inner_fit_end=48,
        economic_prior_inner_coefficient=inner_coef.tolist(),
        lookback=LOOKBACK,horizon=HORIZON,
        epoch_selection="Train-only 2018Q1–2019Q4",
        full_training="2006Q1–2019Q4 only",
        development="2020Q1–2022Q4",
        target_definition_modified=False,
        test_opened=False)
    for folder in (paths["root"],paths["metrics"],paths["predictions"],
                   paths["checkpoints"],paths["plots"]):
        folder.mkdir(parents=True,exist_ok=True)
    existing=(json.loads(paths["manifest"].read_text(encoding="utf-8"))
              if paths["manifest"].is_file() else None)
    if existing is not None and existing.get("signature")!=signature:
        raise RuntimeError(
            "Full 3-mode output has a DIFFERENT source/data/config. "
            "Do not overwrite old checkpoints: archive that run first.")
    train_grid.to_csv(
        paths["root"]/"train_structural_policy_grid.csv",
        index=False,encoding="utf-8-sig")
    graph_panel=panel
    return dict(
        panel=panel,signature=signature,registry=reg,per_mode=per_mode,
        jobs=[job for mode in STAGE04_OUTPUT_MODES for job in per_mode[mode]],
        structure=structured,structure_inner=inner_structured,paths=paths,
        existing=existing,
        graph_panel=graph_panel)

def _full_fold(cache,panel,struct,fit_end,first,last,scaler,missing):
    """Reusable, strictly TRAIN-fitted policy data without Dev-owned stats."""
    key=(fit_end,first,last,scaler,missing)
    if key not in cache:
        cache[key]=prepare_three_axis_fold(
            panel,struct,fit_end,first,last,scaler,missing)
    return cache[key]

def _full_result_record(job,forecast,train_inner,fit_epochs,status="PASS"):
    mode,model,scaler,missing,seed=job
    if forecast.shape!=(DEV_COUNT,NODE_COUNT,3) or not np.isfinite(forecast).all():
        raise RuntimeError(f"Bad forecast shape/finite values {_full_job_id(job)}")
    return dict(
        run_id=_full_job_id(job),status=status,
        prediction_mode=mode,model=model,
        family=("GNN" if model in GNN_MODELS else
                "temporal" if model in (
                    *NEURAL_BASELINES,*EXTRA_NEURAL_BASELINES)
                else "naive" if model in (
                    "Persistence","SeasonalNaive",*EXTRA_NAIVE_BASELINES)
                else "classical_ML"),
        scaler=scaler,missing_mode=missing,seed=seed,
        train_only_inner_mae=(
            float(train_inner) if train_inner is not None else None),
        train_epochs=int(fit_epochs),
    )

def _full_cache_record(job,signature,paths):
    tag=_full_job_id(job)
    mfile=paths["metrics"]/f"{tag}.json"
    pfile=paths["predictions"]/f"{tag}.npz"
    if mfile.is_file() != pfile.is_file():
        raise RuntimeError(f"Partial or corrupted checkpoint {tag}; do not mix runs")
    if not mfile.is_file():
        return None
    row=json.loads(mfile.read_text(encoding="utf-8"))
    if (row.get("run_id")!=tag or row.get("status")!="PASS"
            or row.get("run_contract_source_sha256")!=signature["source_sha256"]):
        raise RuntimeError(f"Stale model checkpoint in full grid {tag}")
    with np.load(pfile,allow_pickle=False) as z:
        pred=z["prediction"].astype(np.float64)
    if pred.shape!=(DEV_COUNT,NODE_COUNT,3) or not np.isfinite(pred).all():
        raise RuntimeError(f"Corrupted forecast checkpoint {tag}")
    return row,pred

def _full_one_fit(job,ctx,cache,device):
    mode,name,scaler,missing,seed=job
    panel=ctx["panel"]
    tic=time.perf_counter()
    if (scaler,missing) != fixed_preprocessing_policy(name):
        raise ValueError(f"Noncanonical Stage04 preprocessing for {name}")
    if name in ("Persistence","SeasonalNaive",*EXTRA_NAIVE_BASELINES):
        forecast=naive_full_forecast(panel,name)
        return forecast,None,0,None,0.
    # No preprocessing search. The family-based policy is locked before
    # scores are seen. Trees with native NaN support receive none/native.
    inner=_full_fold(
        cache,panel,ctx["structure_inner"],48,48,56,scaler,missing)
    full=_full_fold(
        cache,panel,ctx["structure"],56,56,68,scaler,missing)
    if name in (*GNN_MODELS,*NEURAL_BASELINES,*EXTRA_NEURAL_BASELINES):
        selection=fit_three_axis_gnn(
            panel,inner,name,seed,device,None,prediction_mode=mode)
        fitted=fit_three_axis_gnn(
            panel,full,name,seed,device,selection["best_epoch"],
            prediction_mode=mode)
        record=dict(
            model_state_dict=fitted["state_dict"],
            model=name,prediction_mode=mode,seed=seed,
            scaler=scaler,missing_mode=missing,
            epoch=selection["best_epoch"],
            inference_regime=fitted["inference_regime"],
            inference_draws=fitted["inference_draws"],
            predictive_mc_std=fitted.get("predictive_mc_std"),
            predictive_mc_var=fitted.get("predictive_mc_var"),
            signature_sha256=ctx["signature"]["source_sha256"])
        return (fitted["prediction"],selection["train_only_validation_mae"],
                selection["best_epoch"],record,time.perf_counter()-tic)
    # Non-GNN ML baseline: fit Train-internal early window for transparent
    # fixed-policy inner diagnostics and refit the same recipe on FULL Train.
    diagnostic=fit_full_ml(panel,inner,name,seed,mode,missing)
    diagnostic_mae=float(
        np.abs(diagnostic["prediction"]-panel.y[48:56]).mean())
    full_result=fit_full_ml(panel,full,name,seed,mode,missing)
    return (full_result["prediction"],diagnostic_mae,0,
            dict(ml_fitted_estimators=full_result["models"],
                 model=name,prediction_mode=mode,seed=seed,
                 scaler=scaler,missing_mode=missing),
            time.perf_counter()-tic)

def _full_save_one(ctx,job,forecast,inner,epochs,checkpoint,seconds):
    import joblib
    tag=_full_job_id(job)
    paths=ctx["paths"]
    metricfile=paths["metrics"]/f"{tag}.json"
    predfile=paths["predictions"]/f"{tag}.npz"
    folder=paths["checkpoints"]
    if checkpoint:
        if "model_state_dict" in checkpoint:
            target=folder/f"{tag}.pt"
            tmp=target.with_suffix(".pt.writing")
            # Predictive uncertainty lives in the NPZ, NOT in the model
            # checkpoint. NumPy arrays inside a torch.save dictionary
            # require unsafe pickle loading under PyTorch 2.6+.
            # Preserve only tensors and primitive metadata here, so ALL
            # new GNN checkpoints load with weights_only=True.
            model_checkpoint={
                key:value for key,value in checkpoint.items()
                if key not in ("predictive_mc_std","predictive_mc_var")}
            torch.save(model_checkpoint,tmp)
            tmp.replace(target)
        elif "ml_fitted_estimators" in checkpoint:
            target=folder/f"{tag}.joblib"
            tmp=target.with_suffix(".joblib.writing")
            joblib.dump(checkpoint,tmp,compress=3)
            tmp.replace(target)
    row=_full_result_record(job,forecast,inner,epochs)
    row.update(
        signature_source_sha256=ctx["signature"]["source_sha256"],
        run_contract_source_sha256=ctx["signature"]["source_sha256"],
        core_model_family_member=_full_original_job(job,ctx["registry"]),
        preprocessing_fixed_before_dev=True,
        wall_seconds=float(seconds),
        **three_axis_evaluation(forecast,ctx["panel"]))
    if checkpoint is not None and "inference_regime" in checkpoint:
        row["inference_regime"]=checkpoint["inference_regime"]
        row["inference_draws"]=int(checkpoint["inference_draws"])
    # Metrics are written LAST; an interrupted prediction-only write is not
    # validated and will be safely replaced by a complete retry.
    saved_prediction=dict(prediction=forecast.astype(np.float64))
    if checkpoint is not None and checkpoint.get("predictive_mc_std") is not None:
        stdev=np.asarray(checkpoint["predictive_mc_std"],np.float64)
        variance=np.asarray(checkpoint["predictive_mc_var"],np.float64)
        if (stdev.shape!=forecast.shape or variance.shape!=forecast.shape
                or not np.allclose(variance,stdev**2,rtol=1e-10,atol=1e-12)
                or not np.isfinite(variance).all()):
            raise RuntimeError("Invalid B-MTGNN posterior predictive dispersion")
        saved_prediction.update(predictive_mc_std=stdev,
                                predictive_mc_var=variance)
        row["mean_predictive_mc_std"]=float(stdev.mean())
    _atomic_npz(predfile,**saved_prediction)
    _atomic_json(metricfile,row)
    return row

def _full_three_heads_plot(ctx,summary,banks,curves):
    """FOUR concise figures; Persistence in EVERY actual-value line chart."""
    folder=ctx["paths"]["plots"]
    panel=ctx["panel"]
    truth=panel.y[TRAIN_COUNT:TOTAL_COUNT]
    previous=past_scores(panel,np.arange(TRAIN_COUNT,TOTAL_COUNT))
    q=panel.quarters[TRAIN_COUNT:TOTAL_COUNT]
    modes=STAGE04_OUTPUT_MODES
    output=[]
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(folder/name,dpi=160,bbox_inches="tight")
        plt.close(fig)
        output.append(name)
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(len(modes))
    width=.25
    for i,key in enumerate(
            ("best_single_gnn_mae","gnn_ensemble_mae",
             "best_non_gnn_mae")):
        ax.bar(x+(i-1)*width,[
            summary.loc[summary["mode"].eq(mode),key].iloc[0]
            for mode in modes],width,label=key.replace("_"," "))
    ax.set_xticks(x,[v.title() for v in modes])
    ax.set(ylabel="Dev 3-axis MAE, same Stage03 score",
           title="Direct / Residual / Structural forecast comparison")
    ax.grid(axis="y",alpha=.2)
    ax.legend(fontsize=8)
    save(fig,"01_three_modes_model_selection.png")

    columns=("Profitability","Productivity","Growth","ICI")
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    for j,ax in enumerate(axes.flat):
        def levels(v):
            return (v.mean(-1) if j==3 else v[...,j]).mean(1)
        ax.plot(q,levels(truth),"k--",lw=2.4,label="Actual")
        ax.plot(q,levels(previous),ls=":",color="gray",lw=1.8,
                label="Persistence")
        for mode in modes:
            ax.plot(q,levels(banks[mode]),lw=1.9,
                    label=f"{mode} GNN ensemble")
        ax.set(title=columns[j],ylabel="59-industry mean original score")
        ax.tick_params(axis="x",rotation=35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Observed vs GNN mode ensembles and Persistence")
    save(fig,"02_actual_vs_modes_with_persistence.png")

    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(q,(truth-previous)[...,j].mean(1),"k--",lw=2.4,
                label="Actual change")
        ax.plot(q,np.zeros(len(q)),ls=":",color="gray",lw=1.6,
                label="Persistence (zero)")
        for mode in modes:
            ax.plot(q,(banks[mode]-previous)[...,j].mean(1),lw=1.9,
                    label=f"{mode} GNN ensemble")
        ax.set(title=f"{columns[j]} next-quarter change",
               ylabel="Mean actual-score change")
        ax.grid(alpha=.2)
        ax.tick_params(axis="x",rotation=35)
        ax.legend(fontsize=8)
    save(fig,"03_change_vs_modes_with_persistence.png")

    fig,ax=plt.subplots(figsize=(11,5))
    for mode in modes:
        curve=curves[mode]
        k=int(summary.loc[
            summary["mode"].eq(mode),"gnn_ensemble_k"].iloc[0])
        ax.plot(curve.k,curve.axis_mae,".-",alpha=.8,label=mode)
        ax.scatter([k],[float(curve.loc[
            curve.k.eq(k),"axis_mae"].iloc[0])],s=90,marker="*")
    ax.set(xlabel="GNN-only distinct model count",
           ylabel="Dev three-axis MAE",
           title="Original correlation-aware greedy + paired-industry 1-SE")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,"04_gnn_ensemble_k_by_mode.png")
    return output

def generate_mode_reports(paths):
    """Regenerate per-mode model and ensemble figures from saved predictions."""
    root=paths["root"]
    ranking=pd.read_csv(root/"model_mode_leaderboard.csv")
    assignments=pd.read_csv(root/"input_policy_assignment.csv")
    if (assignments.duplicated(["mode","model"]).any()
            or not assignments["preprocessing_fixed"].all()):
        raise RuntimeError("Stage04 plots require unique frozen preprocessing")
    summary=pd.read_csv(root/"mode_comparison_summary.csv").set_index("mode")
    outputs={}
    for mode in STAGE04_OUTPUT_MODES:
        images=paths["plots"]/mode.capitalize()
        images.mkdir(parents=True,exist_ok=True)
        entries=ranking.loc[ranking["mode"]==mode].copy()
        entries=entries.sort_values(["axis_mae","ici_mae","model"])
        lock=json.loads((root/f"{mode}_gnn_ensemble_lock.json").read_text(encoding="utf-8"))
        if lock["candidate_family"]!="GNN_ONLY" or any(
                name not in GNN_MODELS for name in lock["models"]):
            raise RuntimeError("A non-GNN appeared in the ensemble")
        with np.load(root/f"{mode}_gnn_ensemble_predictions.npz",allow_pickle=False) as z:
            truth=z["truth"].copy()
            previous=z["previous_observed"].copy()
            ensemble=z["ensemble"].copy()
            models=z["selected_models"].tolist()
        if (models!=lock["models"] or truth.shape!=(DEV_COUNT,NODE_COUNT,3)
                or not np.isclose(np.abs(ensemble-truth).mean(),
                                  lock["selected_mae"],atol=1e-9)):
            raise RuntimeError(f"{mode} plotted ensemble differs from its lock")
        row=summary.loc[mode]
        best_gnn=entries.loc[entries["model"]==row.best_single_gnn].iloc[0]
        forecasts=[]
        for seed in DEFAULT_SEEDS:
            tag=_full_job_id((mode,best_gnn["model"],best_gnn["scaler"],
                              best_gnn["missing_mode"],seed))
            with np.load(paths["predictions"]/f"{tag}.npz",
                         allow_pickle=False) as z:
                forecasts.append(z["prediction"].copy())
        single=np.mean(forecasts,axis=0)
        best_non_gnn=entries.loc[entries["model"]==row.best_non_gnn_model].iloc[0]
        if not np.isclose(np.abs(single-truth).mean(),
                          best_gnn["axis_mae"],atol=1e-8):
            raise RuntimeError(f"{mode} single-model and ranking mismatch")
        path=pd.read_csv(root/f"{mode}_gnn_ensemble_path.csv")
        if (int(path.selected_k.iloc[0])!=lock["model_count"]
                or not np.isclose(path.loc[
                    path.k==lock["model_count"],"axis_mae"].iloc[0],
                    lock["selected_mae"],atol=1e-9)):
            raise RuntimeError(f"{mode} ensemble selection path mismatch")

        def save(fig,name):
            fig.tight_layout()
            target=images/name
            temporary=images/f"{name}.writing.png"
            fig.savefig(temporary,dpi=160,bbox_inches="tight")
            plt.close(fig)
            if (target.is_file() and
                    hashlib.sha256(target.read_bytes()).digest() ==
                    hashlib.sha256(temporary.read_bytes()).digest()):
                temporary.unlink()
            else:
                temporary.replace(target)
        fig,axes=plt.subplots(1,2,figsize=(18,15))
        for ax,subset,title in (
                (axes[0],entries.loc[entries.family=="GNN"],"41 GNN models"),
                (axes[1],entries.loc[entries.family!="GNN"],"Non-GNN baselines")):
            subset=subset.sort_values("axis_mae",ascending=True)
            color=[
                ("tab:green" if x in lock["models"] else "tab:blue")
                if title.startswith("41") else
                ("tab:orange" if x==row.best_non_gnn_model else "0.65")
                for x in subset.model]
            ax.barh(subset.model,subset.axis_mae,color=color)
            ax.axvline(float(np.abs(previous-truth).mean()),color="black",
                       ls=":",lw=1.5,label="Persistence MAE")
            ax.invert_yaxis()
            ax.set(xlabel="Dev 3-axis MAE (lower is better)",title=title)
            ax.grid(axis="x",alpha=.2)
            ax.tick_params(axis="y",labelsize=8)
            ax.legend(loc="lower right")
        fig.suptitle(
            f"{mode.title()}: {len(entries)} model benchmarks (green = ensemble member)")
        save(fig,"01_model_comparison.png")

        fig,axes=plt.subplots(1,2,figsize=(14,5))
        k=lock["model_count"]
        for ax,metric,title in (
                (axes[0],"axis_mae","Three-axis MAE"),
                (axes[1],"ici_mae","ICI MAE")):
            ax.plot(path.k,path[metric],".-",markersize=3)
            ax.scatter([k],[path.loc[path.k==k,metric].iloc[0]],
                       marker="*",s=180,color="tab:red",zorder=3,
                       label=f"Selected K={k}")
            ax.set(xlabel="Distinct GNN members",ylabel="MAE",title=title)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(f"{mode.title()}: correlation-aware GNN-only selection + 1-SE")
        save(fig,"02_ensemble_selection.png")

        fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
        quarters=pd.period_range("2020Q1","2022Q4",freq="Q").astype(str)
        for j,ax in enumerate(axes.flat):
            def series(a):
                values=a.mean(-1) if j==3 else a[...,j]
                return values.mean(axis=1)
            ax.plot(quarters,series(truth),"k--",lw=2.2,label="Actual")
            ax.plot(quarters,series(previous),":",color="gray",
                    lw=1.8,label="Persistence")
            ax.plot(quarters,series(single),lw=1.5,label="Best single GNN")
            ax.plot(quarters,series(ensemble),lw=1.8,label="GNN ensemble")
            ax.set(title=("Profitability","Productivity","Growth","ICI")[j],
                   ylabel="59-industry mean score")
            ax.tick_params(axis="x",rotation=40)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{mode.title()}: observed vs predictions and Persistence")
        save(fig,"03_actual_vs_predictions_with_persistence.png")

        fig,ax=plt.subplots(figsize=(11,5))
        names=("Profitability","Productivity","Growth","ICI")
        def errors(pred):
            diff=np.abs(pred-truth).mean(axis=(0,1))
            ici=np.abs((pred-truth).mean(-1)).mean()
            return np.r_[diff,ici]
        scores=[
            ("Persistence",errors(previous)),
            ("Best single GNN",errors(single)),
            ("GNN ensemble",errors(ensemble)),
            (f"Best non-GNN ({row.best_non_gnn_model})",
             best_non_gnn[["profitability_mae","productivity_mae",
                           "growth_mae","ici_mae"]].to_numpy(dtype=float))]
        x=np.arange(4)
        for i,(name,values) in enumerate(scores):
            ax.bar(x+(i-1.5)*.19,values,.19,label=name)
        ax.set_xticks(x,names)
        ax.set(ylabel="Dev MAE (lower is better)",
               title=f"{mode.title()}: three Targets and ICI")
        ax.grid(axis="y",alpha=.2)
        ax.legend(fontsize=9)
        save(fig,"04_axis_ici_comparison.png")
        outputs[mode]=[str((images/name).relative_to(root)).replace("\\","/")
                       for name in (
                           "01_model_comparison.png",
                           "02_ensemble_selection.png",
                           "03_actual_vs_predictions_with_persistence.png",
                           "04_axis_ici_comparison.png")]
    report=root/"selected_mode_and_ensemble.json"
    document=json.loads(report.read_text(encoding="utf-8"))
    document["per_mode_plots"]=outputs
    _atomic_json(report,document)
    return outputs

def _full_postprocess(ctx):
    paths=ctx["paths"]
    panel=ctx["panel"]
    truth=panel.y[TRAIN_COUNT:TOTAL_COUNT]
    trials=[]
    policy_rows=[]
    selected_banks={}
    selected_input_policies={}
    for mode in STAGE04_OUTPUT_MODES:
        models=sorted(set(job[1] for job in ctx["per_mode"][mode]))
        for name in models:
            by_policy={}
            for job in ctx["per_mode"][mode]:
                if job[1]!=name:
                    continue
                fetched=_full_cache_record(job,ctx["signature"],paths)
                if fetched is None:
                    raise RuntimeError(f"Missing finished run {_full_job_id(job)}")
                row,pred=fetched
                trials.append(row)
                by_policy.setdefault((job[2],job[3]),[]).append((job[4],pred))
            if len(by_policy)!=1:
                raise RuntimeError(f"Stage04 cannot compare preprocessing for {mode}/{name}")
            (scaler,missing),pairs=next(iter(by_policy.items()))
            if (scaler,missing)!=fixed_preprocessing_policy(name):
                raise RuntimeError(f"Wrong frozen preprocessing for {mode}/{name}")
            pairs=sorted(pairs,key=lambda value:value[0])
            if name in NAIVE_MODELS:
                if len(pairs)!=1:
                    raise RuntimeError("Naive control should appear once")
                bank=np.repeat(pairs[0][1][None],len(DEFAULT_SEEDS),axis=0)
            else:
                if [seed for seed,_ in pairs]!=DEFAULT_SEEDS:
                    raise RuntimeError(f"Unmatched seeds {mode}/{name}")
                bank=np.stack([prediction for _,prediction in pairs])
            predicted=bank.mean(axis=0)
            scores=three_axis_evaluation(predicted,panel)
            best=dict(
                mode=mode,model=name,
                family=("GNN" if name in GNN_MODELS else
                        "temporal" if name in (*NEURAL_BASELINES,*EXTRA_NEURAL_BASELINES)
                        else "naive" if name in NAIVE_MODELS else "classical_ML"),
                scaler=scaler,missing_mode=missing,
                n_seeds=len(pairs),preprocessing_fixed=True,
                preprocessing_selection="PREDEFINED_MODEL_FAMILY_NOT_DEV",
                **scores)
            selected_banks[(mode,name)]=bank
            selected_input_policies[(mode,name)]=(
                best["scaler"],best["missing_mode"])
            policy_rows.append(best)
    pd.DataFrame(trials).to_csv(
        paths["root"]/"full_3mode_trials.csv",
        index=False,encoding="utf-8-sig")
    pd.DataFrame(policy_rows).to_csv(
        paths["root"]/"input_policy_assignment.csv",
        index=False,encoding="utf-8-sig")
    ranking=pd.DataFrame(policy_rows)
    ranking.sort_values(
        ["mode","axis_mae","ici_mae","model"]).to_csv(
        paths["root"]/"model_mode_leaderboard.csv",
        index=False,encoding="utf-8-sig")

    rows=[]
    ensembles={}
    curves={}
    for mode in STAGE04_OUTPUT_MODES:
        graph_ranking=ranking.loc[
            (ranking["mode"]==mode)&(ranking["family"]=="GNN")].sort_values(
                ["axis_mae","ici_mae","model"])
        if len(graph_ranking)!=41:
            raise RuntimeError("All 41 GNNs must be present in each output mode")
        pool={}
        dedup=[]
        for name in graph_ranking["model"]:
            z=selected_banks[(mode,name)]
            duplicate=next((key for key,value in pool.items()
                            if np.array_equal(z,value)),None)
            if duplicate is None:
                pool[name]=z
            else:
                dedup.append(dict(model=name,identical_to=duplicate))
        if not pool or not set(pool).issubset(set(GNN_MODELS)):
            raise RuntimeError("Non-GNN in ensemble candidate registry")
        path,audit,corr,selected=ensemble_selection_path(pool,truth)
        ensemble=np.stack([pool[name].mean(0) for name in selected]).mean(0)
        best_one=min(pool,key=lambda n:(
            float(np.abs(pool[n].mean(0)-truth).mean()),n))
        baseline_rows=ranking.loc[
            (ranking["mode"]==mode)&(ranking["family"]!="GNN")].sort_values(
                ["axis_mae","model"])
        best_baseline=baseline_rows.iloc[0]
        scores=three_axis_evaluation(ensemble,panel)
        alone=three_axis_evaluation(pool[best_one].mean(0),panel)
        rows.append(dict(
            mode=mode,best_single_gnn=best_one,
            best_single_gnn_mae=alone["axis_mae"],
            gnn_ensemble_models="|".join(selected),
            gnn_ensemble_k=len(selected),
            gnn_ensemble_mae=scores["axis_mae"],
            gnn_ensemble_ici_mae=scores["ici_mae"],
            best_non_gnn_model=str(best_baseline["model"]),
            best_non_gnn_mae=float(best_baseline["axis_mae"]),
            deduplicated_gnn_count=len(dedup)))
        ensembles[mode]=ensemble
        curves[mode]=path
        path.to_csv(paths["root"]/f"{mode}_gnn_ensemble_path.csv",
                    index=False,encoding="utf-8-sig")
        audit.to_csv(paths["root"]/f"{mode}_gnn_ensemble_audit.csv",
                     index=False,encoding="utf-8-sig")
        corr.to_csv(paths["root"]/f"{mode}_gnn_residual_correlations.csv",
                    encoding="utf-8-sig")
        _atomic_npz(
            paths["root"]/f"{mode}_gnn_ensemble_predictions.npz",
            ensemble=ensemble,
            truth=truth,previous_observed=past_scores(panel,np.arange(TRAIN_COUNT,TOTAL_COUNT)),
            selected_models=np.asarray(selected))
        _atomic_json(
            paths["root"]/f"{mode}_gnn_ensemble_lock.json",
            dict(status="LOCKED",mode=mode,candidate_family="GNN_ONLY",
                 models=selected,model_count=len(selected),
                 selected_mae=scores["axis_mae"],
                 selected_ici_mae=scores["ici_mae"],
                 equal_weights=[1./len(selected)]*len(selected),
                 identical_prediction_duplicates_removed=dedup,
                 input_policies={
                     n:dict(scaler=selected_input_policies[(mode,n)][0],
                            missing=selected_input_policies[(mode,n)][1])
                     for n in selected},
                  selection_method="Correlation-aware "
                                   "greedy + smallest K within paired "
                                   "industry 1-SE of full-path minimum",
                  test_opened=False))
    summary=pd.DataFrame(rows)
    summary.to_csv(paths["root"]/"mode_comparison_summary.csv",
                   index=False,encoding="utf-8-sig")
    plots=_full_three_heads_plot(ctx,summary,ensembles,curves)
    winner=min(rows,key=lambda r:(r["gnn_ensemble_mae"],
                                  r["gnn_ensemble_ici_mae"],
                                  STAGE04_OUTPUT_MODES.index(r["mode"])))
    mode=winner["mode"]
    lock=json.loads((paths["root"]/f"{mode}_gnn_ensemble_lock.json")
                    .read_text(encoding="utf-8"))
    lock.update(stage="04",model_family="GNN_ONLY",prediction_mode=mode,
                seeds=DEFAULT_SEEDS,target_definition_modified=False,
                input_sha256={key:ctx["signature"][value] for key,value in (
                    ("Data/03_integrated_dataset/raw_train.csv","train_sha256"),
                    ("Data/03_integrated_dataset/raw_val.csv","dev_sha256"))},
                selection_objective="three_axis_dev_MAE_of_five_seed_mean_forecast")
    _atomic_json(paths["root"]/"selected_ensemble_pool.json",lock)
    ensemble_seeds=np.stack([
        selected_banks[(mode,name)] for name in lock["models"]]).mean(0)
    _atomic_npz(paths["root"]/"ensemble_predictions.npz",
                seed_predictions=ensemble_seeds,
                prediction=ensemble_seeds.mean(0),truth=truth,
                previous_observed=past_scores(panel,np.arange(TRAIN_COUNT,TOTAL_COUNT)),
                selected_models=np.asarray(lock["models"]))
    _atomic_json(paths["root"]/"selected_mode_and_ensemble.json",dict(
        status="COMPLETE_ALL_THREE_MODES",
        selected_mode=mode,selected_ensemble_models=lock["models"],
        mode_reports=rows,
        same_stage03_target_values_all_modes=True,
        exactly_one_preprocessing_policy_per_model=True,
        preprocessing_policy=STAGE04_PREPROCESSING_POLICY,
        preprocessing_selected_on_dev=False,
        all_gnn_architectures_each_mode=True,
        ensemble_gnn_only=True,
        model_selection_on_reused_dev=True,
        no_independent_test_claim=True,
        unavailable_optional_baselines=ctx["registry"][
            "optional_ml_unavailable"],
        plots=plots,test_opened=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    generate_mode_reports(paths)
    print("04_FULL_3MODE_ALL_RESULTS_READY",rows,flush=True)
    return rows

def run_full_three_modes(args):
    """Run each model with ONE fixed family policy, three output modes."""
    ctx=full_grid_setup(args)
    if ctx is None:
        return
    import traceback
    paths=ctx["paths"]
    registry=ctx["registry"]
    alljobs=ctx["jobs"]
    signature=ctx["signature"]
    torch.set_num_threads(4)
    # Work through all three modes for each architecture+seed so
    # interruptions never leave only Direct (and no Residual) validated.
    def sortkey(job):
        mode,name,scaler,missing,seed=job
        early=[
            "Persistence","SeasonalNaive","HistoricalMean","LinearDrift",
            "B-MTGNN","MTGNN","AGCRN","GCN-TCN",
            "GraphSAGE-TCN","ARMA-GRU","GATv2-TCN",
            "GraphTransformer-TCN","LightGBM","Ridge","ElasticNet"]
        if name in early:
            ix=early.index(name)
        elif name in GNN_MODELS:
            ix=len(early)+GNN_MODELS.index(name)
        elif name in (*NEURAL_BASELINES,*EXTRA_NEURAL_BASELINES):
            ix=100+(*NEURAL_BASELINES,*EXTRA_NEURAL_BASELINES).index(name)
        else:
            ix=200+sorted(
                set(registry["extra_ml"]+registry["optional_ml_installed"])
            ).index(name)
        return (ix,scaler,missing,seed,
                STAGE04_OUTPUT_MODES.index(mode))
    alljobs.sort(key=sortkey)
    stored=ctx["existing"]
    core_count=sum(_full_original_job(job,registry)
                   for job in ctx["per_mode"]["direct"])
    manifest=dict(
        status="RUNNING",signature=signature,
        started_utc=(stored or {}).get(
            "started_utc",datetime.now(timezone.utc).isoformat()),
        planned_runs=len(alljobs),completed_runs=0,
        error_runs=0,
        test_opened=False,
        continuation="Run python Sources/04_gnn_benchmark.py again",
        preprocessing_policy=STAGE04_PREPROCESSING_POLICY,
        preprocessing_selected_on_dev=False,
        total_core_family_runs=core_count*len(STAGE04_OUTPUT_MODES),
        total_additive_baselines=len(alljobs)-core_count*len(STAGE04_OUTPUT_MODES),
        all_variants_have_same_y_t=True)
    if stored and "source_migration" in stored:
        manifest["source_migration"]=stored["source_migration"]
    _atomic_json(paths["manifest"],manifest)
    tic=time.perf_counter()
    cache={}
    count=0
    failed=[]
    for job in alljobs:
        tag=_full_job_id(job)
        saved=_full_cache_record(job,signature,paths)
        if saved is not None:
            count+=1
            if count%100==0:
                print(f"04_FULL_CACHE_RESTORED [{count}/{len(alljobs)}]",
                      flush=True)
            continue
        if time.perf_counter()-tic > args.time_budget_minutes*60:
            manifest.update(
                status="PARTIAL_TIME_BUDGET",completed_runs=count,
                error_runs=len(failed),errors=failed[:100],
                next_run=tag,
                elapsed_seconds=time.perf_counter()-tic)
            _atomic_json(paths["manifest"],manifest)
            print(f"04_FULL_PARTIAL {count}/{len(alljobs)} "
                  f"failed={len(failed)} next={tag}",flush=True)
            return
        try:
            prediction,inner,epochs,checkpoint,seconds=_full_one_fit(
                job,ctx,cache,args.device)
            record=_full_save_one(
                ctx,job,prediction,inner,epochs,checkpoint,seconds)
            count+=1
            if (count<15 or count%5==0 or
                    job[1] in ("Persistence","SeasonalNaive",*EXTRA_NAIVE_BASELINES)):
                print(
                    f"04_FULL [{count:04d}/{len(alljobs)}] {tag} "
                    f"DevMAE={record['axis_mae']:.5f} "
                    f"TrainInner={record['train_only_inner_mae']} "
                    f"elapsed={seconds:.1f}s",flush=True)
        except Exception as exc:
            error=dict(run_id=tag,type=type(exc).__name__,
                       error=str(exc),trace=traceback.format_exc())
            failed.append(dict(run_id=tag,error=str(exc)))
            _atomic_json(
                paths["metrics"]/f"{tag}.error.json",error)
            print(f"04_FULL_FAILURE {tag} {error['type']}: {exc}",flush=True)
            # Continue other model/policy/mode jobs. Final status must
            # NOT be PASS if even one required baseline failed.
        if (count+len(failed))%10==0:
            manifest.update(
                completed_runs=count,error_runs=len(failed),
                last_job=tag,elapsed_seconds=time.perf_counter()-tic)
            _atomic_json(paths["manifest"],manifest)
    manifest.update(
        completed_runs=count,error_runs=len(failed),
        errors=failed[:200],
        elapsed_seconds=time.perf_counter()-tic,
        training_completed_utc=datetime.now(timezone.utc).isoformat())
    if failed or count!=len(alljobs):
        manifest["status"]="COMPLETED_WITH_FAILURES"
        _atomic_json(paths["manifest"],manifest)
        print(f"04_FULL_WITH_FAILURES good={count} failed={len(failed)} "
              f"expected={len(alljobs)}",flush=True)
        return
    # Check upstream data are unchanged before publishing final selection.
    expected=signature
    for file,name in (
        (ROOT/"Data/03_integrated_dataset/raw_train.csv","train_sha256"),
        (ROOT/"Data/03_integrated_dataset/raw_val.csv","dev_sha256"),
        (STAGE04_COMPONENT_SOURCE,"economics_source_sha256"),
        (STAGE04_TARGET_BOUNDS,"economics_score_bounds_sha256")):
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected[name]:
            raise RuntimeError(f"Upstream data changed DURING Stage04: {file}")
    summary=_full_postprocess(ctx)
    manifest.update(
        status="PASS",scored_modes=list(STAGE04_OUTPUT_MODES),
        model_and_head_comparison="selected_mode_and_ensemble.json",
        three_mode_results=summary,
        completed_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(paths["manifest"],manifest)
    print(f"04_FULL_ALL_PASS total={count} "
          f"core_family_runs={core_count*len(STAGE04_OUTPUT_MODES)} "
          f"extra_baselines={count-core_count*len(STAGE04_OUTPUT_MODES)} "
          "one_preprocessing_policy_per_model=True "
          "test_unopened=True",flush=True)

def full_grid_status():
    paths=full_grid_paths()
    if not paths["manifest"].is_file():
        print("04_FULL_NOT_STARTED")
        return
    record=json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metrics=paths["metrics"]
    successful=len(list(metrics.glob("*.json")))
    failures=len(list(metrics.glob("*.error.json")))
    print("04_FULL_STATUS",record.get("status"),
          "metrics_files",successful-failures,
          "failures",failures,
          "planned",record.get("planned_runs"),
          "completed_manifest",record.get("completed_runs"),
          "core_family_runs",record.get("total_core_family_runs"),
          "last_job",record.get("last_job"),
          "next_run",record.get("next_run"))

# ---------- SINGLE MAIN STAGE04 EXECUTION ----------
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-status",action="store_true")
    parser.add_argument("--plots-only",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--time-budget-minutes",type=float,default=24.)
    parser.add_argument("--device",choices=("cuda","cpu"),default=(
                        "cuda" if torch.cuda.is_available() else "cpu"))
    args=parser.parse_args()
    if args.time_budget_minutes<=0:
        parser.error("Positive time budget required")
    if args.full_status:
        full_grid_status()
    elif args.plots_only:
        paths=full_grid_paths()
        manifest=json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if (manifest["status"]!="PASS"
                or manifest["completed_runs"]!=manifest["planned_runs"]):
            raise RuntimeError("All planned three-mode runs must finish before plotting")
        figures=generate_mode_reports(paths)
        print("04_MODE_FIGURES_READY",figures,flush=True)
    else:
        run_full_three_modes(args)
