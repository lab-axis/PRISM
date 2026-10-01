#!/usr/bin/env python3
"""Final PRISM feature-selection stage.

The stage is intentionally self-contained:
  1) with the full 210-group external pool fixed, select one shared external
     value/mask weight on a 0.05 grid using Train rolling-CV + Validation;
  2) lock that weight before any Top-K selection;
  3) rank external base groups with Train-only LightGBM + XGBoost importance;
  4) retrain B-MTGNN for K=0,5,10,... in every Train rolling-CV fold and seed;
  5) retain paired-1SE CV candidates, evaluate only those candidates on the
     designated Validation split, and choose the smallest Validation-noninferior K;
  6) write the final selected feature files and provenance summary.

Test is never used for weight, feature, K, or model-selection decisions.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent.parent
RAW_TRAIN = ROOT / "Data" / "03_integrated_dataset" / "raw_train.csv"
RAW_VAL = ROOT / "Data" / "03_integrated_dataset" / "raw_val.csv"
PREPROCESSED_DIR = ROOT / "Data" / "04_preprocessing"
REGISTRY = ROOT / "Results" / "02_initial_feature_pool" / "initial_feature_registry.csv"
BMTGNN_DIR = (
    ROOT.parent
    / "Evidence-Bounded Multi-Agent Reasoning for Long-Horizon Cyber Foresight"
    / "Comparative_Evaluation"
    / "BMTGNN"
)

OUTPUT_DIR = ROOT / "Data" / "05_feature_selection"
RESULTS_DIR = ROOT / "Results" / "05_feature_selection"
PLOTS_DIR = RESULTS_DIR / "plots"
TOPK_DETAIL_CSV = RESULTS_DIR / "topk_cv_by_fold_seed.csv"
TOPK_ENSEMBLE_CSV = RESULTS_DIR / "topk_cv_ensemble.csv"
TOPK_PRED_DIR = RESULTS_DIR / "topk_prediction_checkpoints"
INPUT_WEIGHT_LOCK = RESULTS_DIR / "input_weight_robust_selection.json"
INPUT_WEIGHT_VAL_CSV = RESULTS_DIR / "input_weight_selection.csv"
INPUT_WEIGHT_VAL_JSON = RESULTS_DIR / "input_weight_selection.json"
INPUT_WEIGHT_CV_CSV = RESULTS_DIR / "input_weight_cv.csv"
INPUT_WEIGHT_CV_JSON = RESULTS_DIR / "input_weight_cv.json"
INPUT_WEIGHT_ROBUST_CSV = RESULTS_DIR / "input_weight_robust_selection.csv"
TOPK_ROBUST_CSV = RESULTS_DIR / "topk_robust_validation_candidates.csv"
TOPK_ROBUST_JSON = RESULTS_DIR / "topk_robust_selection.json"

IDENTIFIERS = ["quarter", "node_key", "node_name"]
TARGET_COMPONENTS = [
    "target_profitability_raw",
    "target_productivity_raw",
    "target_growth_raw",
]
TARGETS = TARGET_COMPONENTS
HISTORY = [
    "history_profitability_raw",
    "history_productivity_raw",
    "history_growth_raw",
]

L = 12
H = 12
SEEDS = [11, 37, 71]
FI_SEED = 37
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 4
TOPK_STEP = 5
ICI_LOSS_WEIGHT = 1.0
EXTERNAL_VALUE_SCALE = 0.05
EXTERNAL_MASK_SCALE = 0.05
WEIGHT_GRID = [round(i * 0.05, 2) for i in range(21)]
BOOTSTRAPS = 20000


def build_topk_grid(group_count: int) -> list[int]:
    return list(range(0, group_count + 1, TOPK_STEP))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Three one-year-spaced rolling origins reduce overlap among 12-quarter outer
# validation horizons while retaining at least nine supervised training windows.
OUTER_SPECS = [
    ("2014Q4", "2015Q1", "2017Q4"),
    ("2015Q4", "2016Q1", "2018Q4"),
    ("2016Q4", "2017Q1", "2019Q4"),
]


@dataclass
class FoldData:
    train_end: str
    val_start: str
    val_end: str
    fi_train_end: str
    fi_val_start: str
    fi_val_end: str


def ensure_dirs(preserve_topk_checkpoint: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TOPK_PRED_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    for path in RESULTS_DIR.glob("*"):
        preserve = preserve_topk_checkpoint and path in {TOPK_DETAIL_CSV, TOPK_ENSEMBLE_CSV}
        preserve_weight_selection = path.name.startswith("input_weight_")
        if path.is_file() and not preserve and not preserve_weight_selection:
            path.unlink()
    for path in PLOTS_DIR.glob("*"):
        if path.is_file():
            path.unlink()


def import_bmtgnn():
    if not (BMTGNN_DIR / "net.py").exists():
        raise FileNotFoundError(BMTGNN_DIR / "net.py")
    sys.path.insert(0, str(BMTGNN_DIR))
    sys.modules.pop("net", None)
    sys.modules.pop("layer", None)
    from net import gtnet  # type: ignore

    return gtnet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(gtnet, in_dim: int, device: torch.device):
    return gtnet(
        True,
        True,
        3,
        56,
        device,
        None,
        dropout=0.4,
        subgraph_size=20,
        node_dim=20,
        dilation_exponential=2,
        conv_channels=4,
        residual_channels=32,
        skip_channels=256,
        end_channels=256,
        seq_length=L,
        in_dim=in_dim,
        out_dim=3 * H,
        layers=1,
        propalpha=0.1,
        tanhalpha=1.0,
        layer_norm_affline=False,
    ).to(device)


def ordered_panel(raw: pd.DataFrame, feature_ids: list[str], train_end: str):
    quarters = raw["quarter"].drop_duplicates().tolist()
    nodes = sorted(raw["node_key"].astype(str).unique().tolist())
    index = pd.MultiIndex.from_product([quarters, nodes], names=["quarter", "node_key"])
    ordered = raw.set_index(["quarter", "node_key"]).reindex(index).reset_index()
    if ordered["node_name"].isna().any():
        raise RuntimeError("Incomplete (quarter,node) grid")

    history = ordered[HISTORY].apply(pd.to_numeric, errors="raise")
    if history.isna().any().any():
        raise RuntimeError("Causal history is incomplete")
    history_values = history.to_numpy(dtype=np.float32)

    if feature_ids:
        values = ordered[feature_ids].apply(pd.to_numeric, errors="raise")
        observed = values.notna()
        raw_external = values.to_numpy(dtype=np.float32)
        masks = observed.to_numpy(dtype=np.float32)
        external = np.concatenate([raw_external, masks], axis=1)
        # Scaling is intentionally deferred to make_windows(), where each sample
        # gets expanding mean/std fitted only through its own forecast origin.
        invalid_count = 0
    else:
        external = np.empty((len(ordered), 0), dtype=np.float32)
        invalid_count = 0

    features = np.concatenate([history_values, external], axis=1).reshape(
        len(quarters), len(nodes), len(HISTORY) + 2 * len(feature_ids)
    )
    targets = ordered[TARGET_COMPONENTS].to_numpy(dtype=np.float32).reshape(
        len(quarters), len(nodes), 3
    )
    return quarters, nodes, features, targets, invalid_count


def training_starts(quarters: list[str], train_end: str) -> list[int]:
    q_to_i = {q: i for i, q in enumerate(quarters)}
    end = q_to_i[train_end]
    return [
        s
        for s in range(len(quarters) - L - H + 1)
        if s + L + H - 1 <= end
    ]


def validation_start(quarters: list[str], val_start: str, val_end: str) -> int:
    q_to_i = {q: i for i, q in enumerate(quarters)}
    s = q_to_i[val_start] - L
    if s < 0 or quarters[s + L + H - 1] != val_end:
        raise RuntimeError(f"Invalid 12-quarter validation horizon: {val_start}..{val_end}")
    return s


def signed_robust_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-component expanding q05/q95 bounds around the neutral zero."""
    lo = np.nanquantile(values, 0.05, axis=(0, 1))
    hi = np.nanquantile(values, 0.95, axis=(0, 1))
    if (
        np.any(~np.isfinite(lo))
        or np.any(~np.isfinite(hi))
        or np.any(lo >= -1e-12)
        or np.any(hi <= 1e-12)
    ):
        raise RuntimeError(f"Invalid robust dynamic bounds: q05={lo}, q95={hi}")
    return lo.astype(np.float32), hi.astype(np.float32)


def signed_robust_scale(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map zero->0.5, q05->0 and q95->1 with tail clipping."""
    scaled = np.where(
        values >= 0.0,
        0.5 + 0.5 * values / hi,
        0.5 + 0.5 * values / np.abs(lo),
    )
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def make_windows(features: np.ndarray, targets: np.ndarray, starts: list[int]):
    xs = []
    ys = []
    for s in starts:
        origin = s + L - 1
        observed_history = features[: origin + 1, :, : len(HISTORY)]
        lo, hi = signed_robust_bounds(observed_history)

        x = features[s : s + L].copy()
        x[:, :, : len(HISTORY)] = signed_robust_scale(
            x[:, :, : len(HISTORY)], lo, hi
        )

        # External feature normalization is also strictly expanding. Layout is
        # [history | raw external values | observed masks]. A feature that has
        # not yet appeared by this origin remains value=0/mask=0 rather than
        # borrowing moments from a later training quarter.
        external_width = x.shape[2] - len(HISTORY)
        if external_width:
            if external_width % 2 != 0:
                raise RuntimeError("External value/mask channel layout is inconsistent")
            p = external_width // 2
            ext_slice = slice(len(HISTORY), len(HISTORY) + p)
            observed_external = features[: origin + 1, :, ext_slice]
            obs = np.isfinite(observed_external)
            count = obs.sum(axis=(0, 1))
            total = np.where(obs, observed_external, 0.0).sum(axis=(0, 1))
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            centered = np.where(obs, observed_external - mean, 0.0)
            var = np.divide(
                (centered * centered).sum(axis=(0, 1)),
                count,
                out=np.zeros_like(total),
                where=count > 0,
            )
            std = np.sqrt(var)
            valid = np.isfinite(mean) & np.isfinite(std) & (std > 1e-12)
            safe_mean = np.where(np.isfinite(mean), mean, 0.0)
            safe_std = np.where(valid, std, 1.0)
            raw_x = x[:, :, ext_slice]
            scaled_x = (raw_x - safe_mean) / safe_std
            scaled_x[~np.isfinite(scaled_x)] = 0.0
            x[:, :, ext_slice] = scaled_x * EXTERNAL_VALUE_SCALE
            mask_slice = slice(len(HISTORY) + p, len(HISTORY) + 2 * p)
            x[:, :, mask_slice] *= EXTERNAL_MASK_SCALE
        xs.append(x.transpose(2, 1, 0))  # C,N,L

        y = targets[s + L : s + L + H].copy()
        y = signed_robust_scale(y, lo, hi)
        ys.append(y.transpose(2, 0, 1).reshape(3 * H, y.shape[1]))
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
    )


def component_mae_tensor(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    # pred/actual: B, 3H, N -> per-sample equal-weight 3-component MAE in score points.
    b = pred.shape[0]
    err = (pred - actual).abs().reshape(b, 3, H, 56)
    return err.mean(dim=(2, 3)).mean(dim=1) * 100.0


def joint_training_loss(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Equal-weight component L1 plus aggregate-ICI L1 in normalized units."""
    component = (pred - actual).abs().mean()
    b = pred.shape[0]
    pred3 = pred.reshape(b, 3, H, 56)
    actual3 = actual.reshape(b, 3, H, 56)
    ici = (pred3.mean(dim=1) - actual3.mean(dim=1)).abs().mean()
    return component + ICI_LOSS_WEIGHT * ici


def fit_fixed_epochs(
    gtnet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    in_dim: int,
    seed: int,
    epochs: int,
    device: torch.device,
):
    set_seed(seed)
    model = build_model(gtnet, in_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(BATCH_SIZE, len(x_train)),
        shuffle=True,
        generator=gen,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = joint_training_loss(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
    return model


def fit_with_internal_early_stop(
    gtnet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    seed: int,
    device: torch.device,
):
    n = len(x_train)
    if n < 4:
        raise RuntimeError(f"Need at least four train windows for inner early stopping; got {n}")
    holdout = max(1, min(3, int(math.ceil(n * 0.20))))
    x_fit, y_fit = x_train[:-holdout], y_train[:-holdout]
    x_stop, y_stop = x_train[-holdout:], y_train[-holdout:]

    set_seed(seed)
    model = build_model(gtnet, x_train.shape[1], device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_fit, y_fit),
        batch_size=min(BATCH_SIZE, len(x_fit)),
        shuffle=True,
        generator=gen,
    )

    best_val = float("inf")
    best_epoch = 1
    bad = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = joint_training_loss(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(
                joint_training_loss(
                    model(x_stop.to(device)).squeeze(-1),
                    y_stop.to(device),
                ).item()
            )
        if val < best_val - 1e-5:
            best_val = val
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    final_model = fit_fixed_epochs(
        gtnet,
        x_train,
        y_train,
        x_train.shape[1],
        seed,
        best_epoch,
        device,
    )
    return final_model, best_epoch, best_val


def group_indices(registry: pd.DataFrame, feature_ids: list[str]) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(feature_ids)}
    p = len(feature_ids)
    offset = len(HISTORY)
    out = {}
    for base, rows in registry.groupby("base_feature", sort=False):
        idx = []
        for feature_id in rows["feature_id"].astype(str):
            j = index[feature_id]
            idx.extend([offset + j, offset + p + j])
        out[str(base)] = np.asarray(sorted(idx), dtype=np.int64)
    return out


def prediction_metrics(pred: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    """Component and aggregate-ICI MAE in score points for already-computed predictions."""
    pred = pred.clamp(0.0, 1.0)
    component = float(component_mae_tensor(pred, actual).mean().item())
    b = pred.shape[0]
    pred3 = pred.reshape(b, 3, H, 56)
    actual3 = actual.reshape(b, 3, H, 56)
    ici = float((pred3.mean(dim=1) - actual3.mean(dim=1)).abs().mean().item() * 100.0)
    return component, ici


def topk_prediction_path(run_signature: str, fold: int, k: int, seed: int) -> Path:
    return TOPK_PRED_DIR / f"{run_signature[:16]}__fold{fold}__k{k}__seed{seed}.npy"


def derive_fold(quarters: list[str], spec: tuple[str, str, str]) -> FoldData:
    train_end, val_start, val_end = spec
    q_to_i = {q: i for i, q in enumerate(quarters)}
    train_end_i = q_to_i[train_end]
    fi_val_end_i = train_end_i
    fi_val_start_i = fi_val_end_i - (H - 1)
    fi_train_end_i = fi_val_start_i - 1
    if fi_train_end_i < 0:
        raise RuntimeError("FI Train period is empty")
    return FoldData(
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        fi_train_end=quarters[fi_train_end_i],
        fi_val_start=quarters[fi_val_start_i],
        fi_val_end=quarters[fi_val_end_i],
    )


def prepare_windows(
    raw: pd.DataFrame,
    feature_ids: list[str],
    scaler_end: str,
    train_end: str,
    val_start: str,
    val_end: str,
):
    quarters, nodes, features, targets, invalid = ordered_panel(raw, feature_ids, scaler_end)
    if len(nodes) != 56:
        raise RuntimeError(f"Expected 56 nodes, found {len(nodes)}")
    starts = training_starts(quarters, train_end)
    vs = validation_start(quarters, val_start, val_end)
    x_train, y_train = make_windows(features, targets, starts)
    x_val, y_val = make_windows(features, targets, [vs])
    return x_train, y_train, x_val, y_val, invalid


def tree_group_importance_train_only(
    raw: pd.DataFrame,
    registry: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank groups from outer-Train only; no validation labels influence FI."""
    try:
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("Stage 05 FI ranking requires lightgbm and xgboost") from exc

    feature_ids = registry["feature_id"].astype(str).tolist()
    xtr, ytr, context_names = build_fi_training_examples(raw, feature_ids, train_end)
    candidate_start = len(context_names)
    candidate_names = feature_ids + [f"mask__{name}" for name in feature_ids]
    model_specs = {
        "lightgbm": lambda: LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=FI_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=FI_SEED,
            n_jobs=4,
            tree_method="hist",
        ),
    }
    model_importance: dict[str, np.ndarray] = {}
    for model_name, factory in model_specs.items():
        comp_fi = []
        for comp in range(3):
            model = factory()
            model.fit(xtr, ytr[:, comp])
            fi = np.asarray(model.feature_importances_, dtype=float)[candidate_start:]
            total = float(fi.sum())
            comp_fi.append(fi / total if total > 0 else np.zeros_like(fi))
        model_importance[model_name] = np.mean(np.stack(comp_fi), axis=0)

    combined = 0.5 * model_importance["lightgbm"] + 0.5 * model_importance["xgboost"]
    fi_meta = pd.concat(
        [
            registry[["feature_id", "base_feature", "scope", "source"]].copy(),
            registry[["feature_id", "base_feature", "scope", "source"]].assign(
                feature_id=lambda x: "mask__" + x["feature_id"].astype(str)
            ),
        ],
        ignore_index=True,
    )
    channel = pd.DataFrame({
        "feature_id": candidate_names,
        "combined_importance": combined,
        "lightgbm_importance": model_importance["lightgbm"],
        "xgboost_importance": model_importance["xgboost"],
    }).merge(fi_meta, on="feature_id", how="left", validate="one_to_one")
    grouped = (
        channel.groupby(["base_feature", "scope", "source"], as_index=False)[
            ["combined_importance", "lightgbm_importance", "xgboost_importance"]
        ]
        .sum()
    )
    total = float(grouped["combined_importance"].sum())
    if total <= 0:
        raise RuntimeError("Train-only tree FI produced zero total importance")
    grouped["combined_importance"] /= total
    grouped = grouped.sort_values(
        ["combined_importance", "base_feature"], ascending=[False, True]
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped, {
        "fi_train_rows": int(len(xtr)),
        "fi_lightgbm_weight": 0.5,
        "fi_xgboost_weight": 0.5,
        "fi_validation_labels_used": False,
    }


def aggregate_tree_importance(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for base, g in df.groupby("base_feature"):
        g = g.sort_values("fold")
        arr = g["combined_importance"].to_numpy(dtype=float)
        row = {
            "base_feature": base,
            "scope": str(g.iloc[0]["scope"]),
            "source": str(g.iloc[0]["source"]),
            "mean_cv_importance": float(arr.mean()),
            "std_cv_importance": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "se_importance": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        }
        for i, value in enumerate(arr, 1):
            row[f"fold_{i}_importance"] = float(value)
        out.append(row)
    return pd.DataFrame(out).sort_values(
        ["mean_cv_importance", "base_feature"], ascending=[False, True]
    ).reset_index(drop=True)


def aggregate_k_curve(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    fold_means = (
        df.groupby(["fold", "k"], as_index=False)[["mae", "ici_mae"]]
        .mean()
    )
    out = []
    for k, g in fold_means.groupby("k"):
        g = g.sort_values("fold")
        arr = g["mae"].to_numpy(dtype=float)
        ici = g["ici_mae"].to_numpy(dtype=float)
        mean_mae = float(arr.mean())
        mean_ici = float(ici.mean())
        out.append({
            "k": int(k),
            "mean_mae": mean_mae,
            "std_mae": float(arr.std(ddof=1)),
            "se_mae": float(arr.std(ddof=1) / math.sqrt(len(arr))),
            "mean_ici_mae": mean_ici,
            "selection_score": mean_mae + mean_ici,
            "std_ici_mae": float(ici.std(ddof=1)),
            "se_ici_mae": float(ici.std(ddof=1) / math.sqrt(len(ici))),
            **{f"fold_{i}_mae": float(v) for i, v in enumerate(arr, 1)},
            **{f"fold_{i}_ici_mae": float(v) for i, v in enumerate(ici, 1)},
        })
    return pd.DataFrame(out).sort_values("k").reset_index(drop=True)


def select_paired_one_se(k_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Choose the smallest K statistically indistinguishable from the raw CV minimum.

    The uncertainty is computed from paired fold-wise score differences against the
    raw-best K, avoiding the overly conservative unpaired 1-SE rule.
    """
    raw_best = k_curve.sort_values(
        ["selection_score", "mean_mae", "mean_ici_mae", "k"]
    ).iloc[0]
    fold_ids = sorted(
        int(c.removeprefix("fold_").removesuffix("_mae"))
        for c in k_curve.columns
        if c.startswith("fold_") and c.endswith("_mae") and not c.endswith("_ici_mae")
    )
    best_scores = np.asarray(
        [raw_best[f"fold_{i}_mae"] + raw_best[f"fold_{i}_ici_mae"] for i in fold_ids],
        dtype=float,
    )
    mean_delta = []
    paired_se = []
    within = []
    for _, row in k_curve.iterrows():
        scores = np.asarray(
            [row[f"fold_{i}_mae"] + row[f"fold_{i}_ici_mae"] for i in fold_ids],
            dtype=float,
        )
        diff = scores - best_scores
        delta = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        mean_delta.append(delta)
        paired_se.append(se)
        within.append(delta <= se + 1e-12)

    annotated = k_curve.copy()
    annotated["mean_delta_vs_raw_best"] = mean_delta
    annotated["paired_se_vs_raw_best"] = paired_se
    annotated["within_paired_1se"] = within
    selected_k = int(annotated.loc[annotated["within_paired_1se"], "k"].min())
    selected = annotated[annotated["k"] == selected_k].iloc[0]
    return annotated, selected, raw_best


def zero_change_baseline_cv(raw: pd.DataFrame, folds: list[FoldData]) -> pd.DataFrame:
    rows = []
    for fold_idx, fold in enumerate(folds, 1):
        _, _, _, yv, _ = prepare_windows(
            raw,
            [],
            fold.train_end,
            fold.train_end,
            fold.val_start,
            fold.val_end,
        )
        pred = torch.full_like(yv, 0.5)
        mae, ici = prediction_metrics(pred, yv)
        rows.append({
            "fold": fold_idx,
            "mae": mae,
            "ici_mae": ici,
            "selection_score": mae + ici,
        })
    return pd.DataFrame(rows)


def topk_channel_indices(ranking: list[str], groups: dict[str, np.ndarray], k: int) -> np.ndarray:
    active = list(range(len(HISTORY)))
    for name in ranking[:k]:
        active.extend(groups[name].tolist())
    return np.asarray(sorted(set(active)), dtype=np.int64)


def retrained_topk_curve(
    gtnet,
    raw: pd.DataFrame,
    feature_ids: list[str],
    groups: dict[str, np.ndarray],
    fold_rankings: dict[int, list[str]],
    folds: list[FoldData],
    device: torch.device,
    k_grid: list[int],
    run_signature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain sampled Top-K subsets and evaluate the actual 3-seed ensemble."""
    previous = pd.DataFrame()
    if TOPK_DETAIL_CSV.exists():
        try:
            previous = pd.read_csv(TOPK_DETAIL_CSV)
        except Exception:
            previous = pd.DataFrame()
    required = {
        "fold", "seed", "k", "mae", "ici_mae", "epoch",
        "model_input_channels", "run_signature",
    }
    if not previous.empty and not required.issubset(previous.columns):
        previous = pd.DataFrame()
    if not previous.empty:
        previous = previous[previous["run_signature"].astype(str) == run_signature].copy()

    allowed_k = set(int(k) for k in k_grid)
    if not previous.empty:
        previous = previous[previous["k"].astype(int).isin(allowed_k)].copy()
        previous = previous[
            previous.apply(
                lambda r: topk_prediction_path(
                    run_signature, int(r["fold"]), int(r["k"]), int(r["seed"])
                ).exists(),
                axis=1,
            )
        ].copy()
    rows = previous.to_dict("records") if not previous.empty else []
    completed = {
        (int(r["fold"]), int(r["seed"]), int(r["k"]))
        for r in rows
        if int(r["k"]) in allowed_k
        and topk_prediction_path(
            run_signature, int(r["fold"]), int(r["k"]), int(r["seed"])
        ).exists()
    }
    ensemble_rows: list[dict] = []

    for fold_idx, fold in enumerate(folds, 1):
        ranking = fold_rankings[fold_idx]
        xtr_full, ytr, xv_full, yv, _ = prepare_windows(
            raw,
            feature_ids,
            fold.train_end,
            fold.train_end,
            fold.val_start,
            fold.val_end,
        )
        for k in k_grid:
            idx = topk_channel_indices(ranking, groups, k)
            xtr = xtr_full[:, idx]
            xv = xv_full[:, idx]
            missing_seeds = [
                seed for seed in SEEDS if (fold_idx, seed, k) not in completed
            ]
            if not missing_seeds:
                continue
            print(
                f"    fold {fold_idx}/{len(folds)} K={k}/{k_grid[-1]} "
                f"channels={len(idx)} seeds={missing_seeds}"
            )
            for seed in missing_seeds:
                model, epoch, _ = fit_with_internal_early_stop(
                    gtnet, xtr, ytr, seed, device
                )
                model.eval()
                with torch.no_grad():
                    pred = model(xv.to(device)).squeeze(-1).clamp(0.0, 1.0).cpu()
                mae, ici_mae = prediction_metrics(pred, yv)
                np.save(
                    topk_prediction_path(run_signature, fold_idx, k, seed),
                    pred.numpy().astype(np.float32),
                )
                rows.append({
                    "fold": fold_idx,
                    "seed": seed,
                    "k": k,
                    "mae": mae,
                    "ici_mae": ici_mae,
                    "epoch": epoch,
                    "model_input_channels": int(xtr.shape[1]),
                    "run_signature": run_signature,
                })
                completed.add((fold_idx, seed, k))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pd.DataFrame(rows).sort_values(["fold", "k", "seed"]).to_csv(
                TOPK_DETAIL_CSV, index=False, encoding="utf-8-sig"
            )

            pred_paths = [
                topk_prediction_path(run_signature, fold_idx, k, seed)
                for seed in SEEDS
            ]
            if not all(path.exists() for path in pred_paths):
                raise RuntimeError(
                    f"Missing seed prediction checkpoint for fold={fold_idx} K={k}"
                )
            ensemble_pred = torch.tensor(
                np.mean([np.load(path) for path in pred_paths], axis=0),
                dtype=torch.float32,
            )
            ensemble_mae, ensemble_ici = prediction_metrics(ensemble_pred, yv)
            ensemble_rows.append({
                "fold": fold_idx,
                "k": k,
                "mae": ensemble_mae,
                "ici_mae": ensemble_ici,
                "selection_score": ensemble_mae + ensemble_ici,
                "model_input_channels": int(xtr.shape[1]),
                "seed_count": len(SEEDS),
                "run_signature": run_signature,
            })
            pd.DataFrame(ensemble_rows).sort_values(["fold", "k"]).to_csv(
                TOPK_ENSEMBLE_CSV, index=False, encoding="utf-8-sig"
            )

    result = pd.DataFrame(rows)
    result = result[result["k"].astype(int).isin(allowed_k)].copy()
    expected = len(folds) * len(SEEDS) * len(k_grid)
    if len(result) != expected:
        raise RuntimeError(f"Incomplete sampled Top-K grid: {len(result)} != {expected}")
    # Reconstruct every ensemble row from checkpointed seed predictions so a
    # resumed run is identical to an uninterrupted run.
    ensemble_rows = []
    for fold_idx, fold in enumerate(folds, 1):
        _, _, xv_full, yv, _ = prepare_windows(
            raw,
            feature_ids,
            fold.train_end,
            fold.train_end,
            fold.val_start,
            fold.val_end,
        )
        ranking = fold_rankings[fold_idx]
        for k in k_grid:
            idx = topk_channel_indices(ranking, groups, k)
            pred_paths = [
                topk_prediction_path(run_signature, fold_idx, k, seed)
                for seed in SEEDS
            ]
            if not all(path.exists() for path in pred_paths):
                raise RuntimeError(
                    f"Incomplete prediction checkpoints for fold={fold_idx} K={k}"
                )
            ensemble_pred = torch.tensor(
                np.mean([np.load(path) for path in pred_paths], axis=0),
                dtype=torch.float32,
            )
            mae, ici_mae = prediction_metrics(ensemble_pred, yv)
            ensemble_rows.append({
                "fold": fold_idx,
                "k": k,
                "mae": mae,
                "ici_mae": ici_mae,
                "selection_score": mae + ici_mae,
                "model_input_channels": int(len(idx)),
                "seed_count": len(SEEDS),
                "run_signature": run_signature,
            })
    ensemble = pd.DataFrame(ensemble_rows).sort_values(["fold", "k"]).reset_index(drop=True)
    ensemble.to_csv(TOPK_ENSEMBLE_CSV, index=False, encoding="utf-8-sig")
    return (
        result.sort_values(["fold", "k", "seed"]).reset_index(drop=True),
        ensemble,
    )


def write_selected_outputs(selected_groups: list[str], registry: pd.DataFrame) -> tuple[dict[str, int], int]:
    group_set = set(selected_groups)
    selected_channels = registry[registry["base_feature"].isin(group_set)]["feature_id"].astype(str).tolist()
    keep = IDENTIFIERS + TARGETS + HISTORY + selected_channels + [f"mask__{c}" for c in selected_channels]
    rows = {}
    for split in ("train", "val", "test"):
        src = PREPROCESSED_DIR / f"preprocessed_{split}.csv"
        dst = OUTPUT_DIR / f"selected_{split}.csv"
        df = pd.read_csv(src, dtype={"quarter": str, "node_key": str, "node_name": str})
        df[keep].to_csv(dst, index=False, encoding="utf-8-sig")
        rows[split] = len(df)
    registry[registry["base_feature"].isin(group_set)].to_csv(
        RESULTS_DIR / "selected_channels.csv", index=False, encoding="utf-8-sig"
    )
    return rows, len(selected_channels)


def save_plots(
    importance: pd.DataFrame,
    k_curve: pd.DataFrame,
    selected_k: int,
    zero_component_mae: float,
    zero_ici_mae: float,
) -> list[str]:
    generated = []

    top = importance.head(30).sort_values("importance")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top["base_feature"], top["importance"], xerr=top["se_importance"], capsize=2)
    ax.set_xlabel("Aggregated normalized FI")
    ax.set_title("Train-only LightGBM + XGBoost feature-group importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = PLOTS_DIR / "01_tree_ensemble_group_importance.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # Breakout-Signals-Filtering-style Top-K performance figure: two validation
    # metrics against sequentially expanded Top-K subsets, with the final K marked.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharex=True)
    kx = k_curve["k"].to_numpy(dtype=float)
    panels = [
        (axes[0], "mean_mae", "se_mae", "Component MAE", "(a) Component MAE", zero_component_mae),
        (axes[1], "mean_ici_mae", "se_ici_mae", "ICI MAE", "(b) ICI MAE", zero_ici_mae),
    ]
    for ax, mean_col, se_col, ylabel, title, baseline in panels:
        mean = k_curve[mean_col].to_numpy(dtype=float)
        se = k_curve[se_col].to_numpy(dtype=float)
        ax.plot(kx, mean, linewidth=1.8)
        ax.fill_between(kx, mean - se, mean + se, alpha=0.18)
        ax.axhline(
            baseline,
            linestyle=":",
            linewidth=1.3,
            label="Zero-change baseline",
        )
        selected_row = k_curve[k_curve["k"] == selected_k].iloc[0]
        ax.scatter([selected_k], [float(selected_row[mean_col])], s=38, zorder=4)
        ax.axvline(selected_k, linestyle="--", linewidth=1.25, label=f"K={selected_k}")
        ax.set_xlabel("Number of top-ranked external feature groups (K)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend()
    fig.suptitle("Top-K B-MTGNN feature-subset performance")
    fig.tight_layout()
    path = PLOTS_DIR / "02_topk_feature_selection.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    generated.append(str(path.relative_to(ROOT)))
    pdf_path = PLOTS_DIR / "02_topk_feature_selection.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    generated.append(str(pdf_path.relative_to(ROOT)))
    plt.close(fig)

    top_stability = importance.head(30)
    fold_cols = [c for c in top_stability.columns if c.startswith("fold_") and c.endswith("_importance")]
    matrix = top_stability[fold_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(top_stability)), top_stability["base_feature"], fontsize=7)
    ax.set_xticks(np.arange(len(fold_cols)), [f"Fold {i}" for i in range(1, len(fold_cols) + 1)])
    ax.set_title("Tree-ensemble FI stability across rolling origins")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Normalized FI")
    fig.tight_layout()
    path = PLOTS_DIR / "03_tree_ensemble_importance_stability.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    compare_ks = [0, selected_k, int(k_curve["k"].max())]
    compare_ks = list(dict.fromkeys(compare_ks))
    compare = k_curve.set_index("k").loc[compare_ks]
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    labels = ["History only" if k == 0 else (f"Selected K={k}" if k == selected_k else f"Full K={k}") for k in compare_ks]
    values = compare["mean_mae"].to_numpy(dtype=float).tolist() + [zero_component_mae]
    errors = compare["se_mae"].to_numpy(dtype=float).tolist() + [0.0]
    labels = labels + ["Zero-change"]
    ax.bar(labels, values, yerr=errors, capsize=5)
    ax.set_ylabel("Rolling-CV MAE")
    ax.set_title("Retrained B-MTGNN Top-K comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = PLOTS_DIR / "04_bmtgnn_topk_comparison.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))
    return generated


def evaluate_ensemble(
    gtnet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float, float, float, list[dict]]:
    preds = []
    epochs = []
    seed_rows = []
    for seed in SEEDS:
        model, epoch, inner = fit_with_internal_early_stop(
            gtnet, x_train, y_train, seed, device
        )
        model.eval()
        with torch.no_grad():
            pred = model(x_eval.to(device)).squeeze(-1).clamp(0.0, 1.0).cpu()
        mae, ici = prediction_metrics(pred, y_eval)
        preds.append(pred.numpy())
        epochs.append(epoch)
        seed_rows.append({
            "seed": int(seed),
            "epoch": int(epoch),
            "inner_stop_loss": float(inner),
            "mae": float(mae),
            "ici_mae": float(ici),
            "score": float(mae + ici),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble_np = np.mean(preds, axis=0)
    ensemble = torch.tensor(ensemble_np, dtype=torch.float32)
    mae, ici = prediction_metrics(ensemble, y_eval)
    return ensemble_np, float(mae), float(ici), float(np.mean(epochs)), seed_rows


def _unit_scale_windows(
    raw: pd.DataFrame,
    feature_ids: list[str],
    scaler_end: str,
    train_end: str,
    val_start: str,
    val_end: str,
):
    global EXTERNAL_VALUE_SCALE, EXTERNAL_MASK_SCALE
    old_value = EXTERNAL_VALUE_SCALE
    old_mask = EXTERNAL_MASK_SCALE
    EXTERNAL_VALUE_SCALE = 1.0
    EXTERNAL_MASK_SCALE = 1.0
    try:
        return prepare_windows(raw, feature_ids, scaler_end, train_end, val_start, val_end)
    finally:
        EXTERNAL_VALUE_SCALE = old_value
        EXTERNAL_MASK_SCALE = old_mask


def _apply_external_weight(x: torch.Tensor, weight: float) -> torch.Tensor:
    out = x.clone()
    out[:, len(HISTORY):] *= float(weight)
    return out


def _evaluate_weight(
    gtnet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    weight: float,
    device: torch.device,
) -> tuple[float, float, float, list[dict]]:
    _, mae, ici, epoch_mean, seeds = evaluate_ensemble(
        gtnet,
        _apply_external_weight(x_train, weight),
        y_train,
        _apply_external_weight(x_eval, weight),
        y_eval,
        device,
    )
    return mae, ici, epoch_mean, seeds


def select_input_weight(
    gtnet,
    raw_train: pd.DataFrame,
    registry: pd.DataFrame,
    folds: list[FoldData],
    device: torch.device,
    fresh: bool = False,
) -> dict:
    """Select the shared value/mask weight before any Top-K selection."""
    if INPUT_WEIGHT_LOCK.exists() and not fresh:
        payload = json.loads(INPUT_WEIGHT_LOCK.read_text(encoding="utf-8"))
        if payload.get("complete") and not payload.get("test_used"):
            return payload

    feature_ids = registry["feature_id"].astype(str).tolist()
    if int(registry["base_feature"].nunique()) != 210 or len(feature_ids) != 948:
        raise RuntimeError("Input-weight selection requires the full 210-group/948-channel pool")

    raw_val = pd.read_csv(
        RAW_VAL, dtype={"quarter": str, "node_key": str, "node_name": str}
    )
    raw_train_val = pd.concat([raw_train, raw_val], ignore_index=True)
    xtr_v, ytr_v, xv_v, yv_v, _ = _unit_scale_windows(
        raw_train_val, feature_ids, "2019Q4", "2019Q4", "2020Q1", "2022Q4"
    )

    val_rows = []
    for i, weight in enumerate(WEIGHT_GRID, 1):
        print(f"[weight val {i}/{len(WEIGHT_GRID)}] {weight:.2f}", flush=True)
        mae, ici, epoch, seeds = _evaluate_weight(
            gtnet, xtr_v, ytr_v, xv_v, yv_v, weight, device
        )
        val_rows.append({
            "weight": weight,
            "epoch_mean": epoch,
            "mae": mae,
            "ici_mae": ici,
            "score": mae + ici,
            "seed_metrics_json": json.dumps(seeds, ensure_ascii=False),
        })
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(INPUT_WEIGHT_VAL_CSV, index=False, encoding="utf-8-sig")
    INPUT_WEIGHT_VAL_JSON.write_text(
        json.dumps({
            "complete": True,
            "test_used": False,
            "weight_grid": WEIGHT_GRID,
            "feature_pool": "all 210 base groups / 948 channels",
            "results": val_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fold_windows = []
    for fold_idx, fold in enumerate(folds, 1):
        xtr, ytr, xv, yv, _ = _unit_scale_windows(
            raw_train, feature_ids, fold.train_end, fold.train_end, fold.val_start, fold.val_end
        )
        fold_windows.append((fold_idx, fold, xtr, ytr, xv, yv))

    cv_rows = []
    for i, weight in enumerate(WEIGHT_GRID, 1):
        print(f"[weight cv {i}/{len(WEIGHT_GRID)}] {weight:.2f}", flush=True)
        for fold_idx, fold, xtr, ytr, xv, yv in fold_windows:
            mae, ici, epoch, _ = _evaluate_weight(
                gtnet, xtr, ytr, xv, yv, weight, device
            )
            cv_rows.append({
                "weight": weight,
                "fold": fold_idx,
                "train_end": fold.train_end,
                "val_period": f"{fold.val_start}..{fold.val_end}",
                "mae": mae,
                "ici_mae": ici,
                "score": mae + ici,
                "epoch_mean": epoch,
            })
    cv_detail = pd.DataFrame(cv_rows)
    cv_detail.to_csv(INPUT_WEIGHT_CV_CSV, index=False, encoding="utf-8-sig")
    cv_mean = (
        cv_detail.groupby("weight", as_index=False)[["mae", "ici_mae", "score"]]
        .mean()
        .rename(columns={"mae": "cv_mae", "ici_mae": "cv_ici_mae", "score": "cv_score"})
    )
    INPUT_WEIGHT_CV_JSON.write_text(
        json.dumps({
            "complete": True,
            "test_used": False,
            "weight_grid": WEIGHT_GRID,
            "results": cv_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    val_mean = val_df[["weight", "mae", "ici_mae", "score"]].rename(
        columns={"mae": "val_mae", "ici_mae": "val_ici_mae", "score": "val_score"}
    )
    joined = cv_mean.merge(val_mean, on="weight", validate="one_to_one")
    joined["cv_regret"] = joined["cv_score"] - joined["cv_score"].min()
    joined["val_regret"] = joined["val_score"] - joined["val_score"].min()
    joined["worst_regret"] = joined[["cv_regret", "val_regret"]].max(axis=1)
    joined["mean_regret"] = joined[["cv_regret", "val_regret"]].mean(axis=1)
    joined["cv_rank"] = joined["cv_score"].rank(method="min").astype(int)
    joined["val_rank"] = joined["val_score"].rank(method="min").astype(int)
    chosen = joined.sort_values(["worst_regret", "mean_regret", "weight"]).iloc[0]
    joined.to_csv(INPUT_WEIGHT_ROBUST_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "complete": True,
        "test_used": False,
        "selection_stage": "before Top-K feature selection",
        "weight_grid_step": 0.05,
        "feature_pool_during_weight_selection": "all 210 base groups / 948 channels",
        "shared_value_mask_weight": True,
        "selection_rule": (
            "minimize worst-case regret across Train rolling-CV and designated Validation; "
            "tie-break by mean regret then smaller weight"
        ),
        "cv_best_weight": float(joined.loc[joined["cv_score"].idxmin(), "weight"]),
        "validation_best_weight": float(joined.loc[joined["val_score"].idxmin(), "weight"]),
        "cv_validation_spearman": float(joined["cv_score"].corr(joined["val_score"], method="spearman")),
        "selected_weight": float(chosen["weight"]),
        "selected_cv_score": float(chosen["cv_score"]),
        "selected_validation_score": float(chosen["val_score"]),
        "selected_worst_regret": float(chosen["worst_regret"]),
        "results": joined.to_dict("records"),
    }
    INPUT_WEIGHT_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _node_score(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    p = pred.reshape(-1, 3, H, 56)
    y = actual.reshape(-1, 3, H, 56)
    component = np.abs(p - y).mean(axis=(0, 1, 2)) * 100.0
    ici = np.abs(p.mean(axis=1) - y.mean(axis=1)).mean(axis=(0, 1)) * 100.0
    return component + ici


def finalize_topk_with_validation(
    gtnet,
    raw_train: pd.DataFrame,
    registry: pd.DataFrame,
    final_ranking: pd.DataFrame,
    k_curve: pd.DataFrame,
    topk_ensemble: pd.DataFrame,
    device: torch.device,
    zero_component_mae: float,
    zero_ici_mae: float,
    summary: dict,
) -> tuple[int, dict]:
    candidates = sorted(
        int(k) for k in k_curve.loc[k_curve["within_paired_1se"].astype(bool), "k"].tolist()
    )
    if not candidates:
        raise RuntimeError("No paired-1SE Top-K candidates")
    raw_val = pd.read_csv(
        RAW_VAL, dtype={"quarter": str, "node_key": str, "node_name": str}
    )
    raw = pd.concat([raw_train, raw_val], ignore_index=True)
    predictions: dict[int, np.ndarray] = {}
    rows = []
    yv_ref = None
    for i, k in enumerate(candidates, 1):
        print(f"[top-k val {i}/{len(candidates)}] K={k}", flush=True)
        groups = final_ranking.head(k)["base_feature"].astype(str).tolist() if k else []
        ids = (
            registry[registry["base_feature"].isin(groups)]["feature_id"].astype(str).tolist()
            if k else []
        )
        xtr, ytr, xv, yv, _ = prepare_windows(
            raw, ids, "2019Q4", "2019Q4", "2020Q1", "2022Q4"
        )
        if yv_ref is None:
            yv_ref = yv
        elif not torch.equal(yv_ref, yv):
            raise RuntimeError("Validation target mismatch across K")
        pred, mae, ici, epoch, seeds = evaluate_ensemble(
            gtnet, xtr, ytr, xv, yv, device
        )
        predictions[k] = pred
        rows.append({
            "k": k,
            "selected_channel_count": len(ids),
            "input_channels": int(xtr.shape[1]),
            "mae": mae,
            "ici_mae": ici,
            "score": mae + ici,
            "epoch_mean": epoch,
            "seed_metrics_json": json.dumps(seeds, ensure_ascii=False),
        })

    table = pd.DataFrame(rows)
    val_best_k = int(table.sort_values(["score", "mae", "ici_mae", "k"]).iloc[0]["k"])
    actual = yv_ref.numpy()
    best_node = _node_score(predictions[val_best_k], actual)
    rng = np.random.default_rng(2601006)
    samples = rng.integers(0, 56, size=(BOOTSTRAPS, 56))
    boot_rows = []
    robust = []
    for k in candidates:
        delta = _node_score(predictions[k], actual) - best_node
        boot = delta[samples].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        noninferior = bool(float(lo) <= 1e-12)
        boot_rows.append({
            "k": k,
            "mean_delta_vs_val_best": float(delta.mean()),
            "bootstrap_ci95_low": float(lo),
            "bootstrap_ci95_high": float(hi),
            "validation_noninferior": noninferior,
        })
        if noninferior:
            robust.append(k)
    table = table.merge(pd.DataFrame(boot_rows), on="k", how="left")
    selected_k = int(min(robust))
    neutral = torch.full_like(yv_ref, 0.5)
    zero_mae, zero_ici = prediction_metrics(neutral, yv_ref)
    table.to_csv(TOPK_ROBUST_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "complete": True,
        "test_used": False,
        "protocol": (
            "Train Top-K paired 1-SE filter -> Validation paired industry bootstrap -> "
            "smallest robust K"
        ),
        "external_value_scale": EXTERNAL_VALUE_SCALE,
        "external_mask_scale": EXTERNAL_MASK_SCALE,
        "train_cv_raw_best_k": int(summary["raw_best_k"]),
        "train_cv_paired_1se_candidates": candidates,
        "validation_best_among_cv_candidates": val_best_k,
        "validation_robust_candidates": robust,
        "selected_k": selected_k,
        "zero_change_validation": {
            "mae": zero_mae,
            "ici_mae": zero_ici,
            "score": zero_mae + zero_ici,
        },
        "bootstrap_repetitions": BOOTSTRAPS,
        "candidate_results": table.to_dict("records"),
    }
    TOPK_ROBUST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = final_ranking.head(selected_k).copy()
    selected.to_csv(RESULTS_DIR / "selected_base_features.csv", index=False, encoding="utf-8-sig")
    selected_groups = selected["base_feature"].astype(str).tolist()
    split_rows, selected_channel_count = write_selected_outputs(selected_groups, registry)
    topk_ensemble[topk_ensemble["k"] == selected_k].to_csv(
        RESULTS_DIR / "bmtgnn_selected_subset_cv.csv", index=False, encoding="utf-8-sig"
    )
    save_plots(
        final_ranking,
        k_curve,
        selected_k,
        zero_component_mae,
        zero_ici_mae,
    )
    cv_row = k_curve[k_curve["k"] == selected_k].iloc[0]
    val_row = table[table["k"] == selected_k].iloc[0]
    summary.update({
        "method": (
            "Robust pre-Top-K input-weight selection + Train-only LightGBM+XGBoost ranking + "
            "retrained B-MTGNN Top-K rolling-CV + Validation bootstrap finalization"
        ),
        "selection_target": (
            "Train rolling-CV paired 1-SE candidate filter followed by designated Validation "
            "paired-industry bootstrap; choose the smallest Validation-noninferior K"
        ),
        "cv_parsimonious_k_before_validation": int(summary["selected_k"]),
        "selected_k": selected_k,
        "topk_robust_selection_artifact": str(TOPK_ROBUST_JSON.relative_to(ROOT)),
        "topk_validation_best_candidate_k": val_best_k,
        "topk_validation_robust_candidates": robust,
        "selected_retrained_cv_mae": float(cv_row["mean_mae"]),
        "selected_retrained_cv_ici_mae": float(cv_row["mean_ici_mae"]),
        "selected_cv_selection_score": float(cv_row["selection_score"]),
        "selected_validation_mae": float(val_row["mae"]),
        "selected_validation_ici_mae": float(val_row["ici_mae"]),
        "selected_validation_selection_score": float(val_row["score"]),
        "zero_change_validation_mae": float(zero_mae),
        "zero_change_validation_ici_mae": float(zero_ici),
        "zero_change_validation_selection_score": float(zero_mae + zero_ici),
        "selected_beats_zero_change_validation": bool(float(val_row["score"]) < zero_mae + zero_ici),
        "selected_channel_count": int(selected_channel_count),
        "selected_model_input_columns": int(len(HISTORY) + selected_channel_count * 2),
        "validation_split_used_for_selection": True,
        "test_split_used_for_selection": False,
        "output_rows": split_rows,
    })
    return selected_k, summary


def main() -> None:
    global EXTERNAL_VALUE_SCALE, EXTERNAL_MASK_SCALE
    parser = argparse.ArgumentParser(description="B-MTGNN sampled Top-K feature selection")
    parser.add_argument(
        "--fresh-weight", action="store_true",
        help="recompute the pre-Top-K 0.05-step input-weight sweep on Train CV + Validation",
    )
    parser.add_argument(
        "--fresh-topk", action="store_true",
        help="discard any saved Top-K fold/seed checkpoint and recompute from K=0",
    )
    args = parser.parse_args()
    if args.fresh_topk:
        for path in (TOPK_DETAIL_CSV, TOPK_ENSEMBLE_CSV):
            if path.exists():
                path.unlink()
        if TOPK_PRED_DIR.exists():
            for path in TOPK_PRED_DIR.glob("*.npy"):
                path.unlink()
    ensure_dirs(preserve_topk_checkpoint=True)
    registry = pd.read_csv(REGISTRY)
    channel_count = len(registry)
    group_count = int(registry["base_feature"].nunique())
    if channel_count <= 0 or group_count <= 0:
        raise RuntimeError("Empty feature registry")
    topk_grid = build_topk_grid(group_count)
    feature_ids = registry["feature_id"].astype(str).tolist()
    groups = group_indices(registry, feature_ids)

    raw = pd.read_csv(RAW_TRAIN, dtype={"quarter": str, "node_key": str, "node_name": str})
    quarters = raw["quarter"].drop_duplicates().tolist()
    folds = [derive_fold(quarters, spec) for spec in OUTER_SPECS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtnet = import_bmtgnn()

    if args.fresh_weight:
        for path in (
            INPUT_WEIGHT_VAL_CSV,
            INPUT_WEIGHT_VAL_JSON,
            INPUT_WEIGHT_CV_CSV,
            INPUT_WEIGHT_CV_JSON,
            INPUT_WEIGHT_ROBUST_CSV,
            INPUT_WEIGHT_LOCK,
        ):
            if path.exists():
                path.unlink()
    weight_lock = select_input_weight(
        gtnet, raw, registry, folds, device, fresh=args.fresh_weight
    )
    locked_weight = float(weight_lock["selected_weight"])
    EXTERNAL_VALUE_SCALE = locked_weight
    EXTERNAL_MASK_SCALE = locked_weight
    print(f"[0/6] Locked external value/mask weight={locked_weight:.2f}")

    importance_rows: list[dict] = []
    fold_rankings: dict[int, list[str]] = {}
    fold_meta = []

    print(f"[1/6] Train-only LightGBM + XGBoost feature ranking; B-MTGNN validation on {device}")
    for fold_idx, fold in enumerate(folds, 1):
        print(
            f"  fold {fold_idx}: outer train <= {fold.train_end}, "
            f"outer val {fold.val_start}..{fold.val_end}"
        )
        fold_fi, fi_metrics = tree_group_importance_train_only(raw, registry, fold.train_end)
        fold_rankings[fold_idx] = fold_fi["base_feature"].tolist()
        fold_rows = fold_fi.copy()
        fold_rows["fold"] = fold_idx
        importance_rows.extend(fold_rows.to_dict("records"))

        fold_meta.append({
            "fold": fold_idx,
            "fi_train_end": fold.train_end,
            "outer_train_end": fold.train_end,
            "outer_val": [fold.val_start, fold.val_end],
            **fi_metrics,
        })

    run_signature = hashlib.sha256(
        (
            file_sha256(RAW_TRAIN)
            + file_sha256(REGISTRY)
            + file_sha256(Path(__file__).resolve())
            + file_sha256(BMTGNN_DIR / "net.py")
            + file_sha256(BMTGNN_DIR / "layer.py")
            + json.dumps(fold_rankings, sort_keys=True)
            + json.dumps(OUTER_SPECS)
            + json.dumps(SEEDS)
            + f"|ici_loss_weight={ICI_LOSS_WEIGHT}|topk_step={TOPK_STEP}|seed_ensemble=v1"
            + f"|external_value_scale={EXTERNAL_VALUE_SCALE}|external_mask_scale={EXTERNAL_MASK_SCALE}"
        ).encode("utf-8")
    ).hexdigest()

    pd.DataFrame(importance_rows).to_csv(
        RESULTS_DIR / "tree_group_importance_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    zero_cv = zero_change_baseline_cv(raw, folds)
    zero_cv.to_csv(
        RESULTS_DIR / "zero_change_baseline_cv.csv", index=False, encoding="utf-8-sig"
    )
    zero_component_mae = float(zero_cv["mae"].mean())
    zero_ici_mae = float(zero_cv["ici_mae"].mean())
    zero_selection_score = float(zero_cv["selection_score"].mean())

    print(f"[2/6] Sampled retrained Top-K curve: K={topk_grid}")
    topk_detail, topk_ensemble = retrained_topk_curve(
        gtnet,
        raw,
        feature_ids,
        groups,
        fold_rankings,
        folds,
        device,
        topk_grid,
        run_signature,
    )
    topk_detail.to_csv(TOPK_DETAIL_CSV, index=False, encoding="utf-8-sig")
    topk_ensemble.to_csv(TOPK_ENSEMBLE_CSV, index=False, encoding="utf-8-sig")

    print("[3/6] Selecting parsimonious Top-K by paired 1-SE rule")
    k_curve = aggregate_k_curve(topk_ensemble.to_dict("records"))
    k_curve, best_row, raw_best_row = select_paired_one_se(k_curve)
    k_curve.to_csv(RESULTS_DIR / "topk_cv_performance.csv", index=False, encoding="utf-8-sig")
    selected_k = int(best_row["k"])
    print(
        f"    raw best K={int(raw_best_row['k'])}; selected K={selected_k}: "
        f"component MAE={best_row['mean_mae']:.4f}, "
        f"ICI MAE={best_row['mean_ici_mae']:.4f}, "
        f"combined={best_row['selection_score']:.4f}"
    )

    selected_cv = topk_ensemble[topk_ensemble["k"] == selected_k].copy()
    selected_cv.to_csv(
        RESULTS_DIR / "bmtgnn_selected_subset_cv.csv", index=False, encoding="utf-8-sig"
    )
    history_cv = topk_ensemble[topk_ensemble["k"] == 0].copy()
    history_cv.to_csv(
        RESULTS_DIR / "bmtgnn_history_only_cv.csv", index=False, encoding="utf-8-sig"
    )
    full_k = topk_grid[-1]
    full_cv = topk_ensemble[topk_ensemble["k"] == full_k].copy()
    full_cv.to_csv(
        RESULTS_DIR / "bmtgnn_full_model_cv.csv", index=False, encoding="utf-8-sig"
    )

    print("[4/6] Final full-Train feature ranking")
    final_tree, final_fi_metrics = tree_group_importance_train_only(raw, registry, "2019Q4")
    stability = aggregate_tree_importance(importance_rows)
    final_ranking = final_tree.rename(columns={"combined_importance": "importance"}).merge(
        stability[
            ["base_feature", "mean_cv_importance", "se_importance"]
            + [c for c in stability.columns if c.startswith("fold_") and c.endswith("_importance")]
        ],
        on="base_feature",
        how="left",
        validate="one_to_one",
    )
    final_ranking = final_ranking.sort_values(
        ["importance", "base_feature"], ascending=[False, True]
    ).reset_index(drop=True)
    final_ranking["rank"] = np.arange(1, len(final_ranking) + 1)
    final_ranking.to_csv(
        RESULTS_DIR / "feature_group_importance.csv", index=False, encoding="utf-8-sig"
    )
    selected = final_ranking.head(selected_k).copy()
    selected.to_csv(
        RESULTS_DIR / "selected_base_features.csv", index=False, encoding="utf-8-sig"
    )

    print("[5/6] Writing selected data and plots")
    selected_groups = selected["base_feature"].tolist()
    split_rows, selected_channel_count = write_selected_outputs(selected_groups, registry)
    plots = save_plots(
        final_ranking,
        k_curve,
        selected_k,
        zero_component_mae,
        zero_ici_mae,
    )

    selected_component = float(best_row["mean_mae"])
    selected_ici = float(best_row["mean_ici_mae"])
    history_row = k_curve[k_curve["k"] == 0].iloc[0]
    full_row = k_curve[k_curve["k"] == full_k].iloc[0]

    summary = {
        "method": "Train-only LightGBM+XGBoost FI ensemble ranking + sampled retrained B-MTGNN Top-K rolling-CV selection",
        "model_source": str(BMTGNN_DIR.relative_to(ROOT.parent)),
        "device": str(device),
        "input_history_quarters": L,
        "forecast_horizon_quarters": H,
        "selection_target": (
            "smallest K within paired 1-SE of the raw minimum outer rolling-CV "
            "(three-component MAE + ICI MAE); both metrics are in score points"
        ),
        "training_loss": "component L1 + ICI L1",
        "ici_loss_weight": ICI_LOSS_WEIGHT,
        "external_value_scale": EXTERNAL_VALUE_SCALE,
        "external_mask_scale": EXTERNAL_MASK_SCALE,
        "input_weight_selection_artifact": str(INPUT_WEIGHT_LOCK.relative_to(ROOT)),
        "input_weight_selection_rule": weight_lock["selection_rule"],
        "always_on_history_columns": HISTORY,
        "seeds": SEEDS,
        "base_feature_groups_total": group_count,
        "channels_total": channel_count,
        "topk_grid": topk_grid,
        "topk_step": TOPK_STEP,
        "normalization_policy": (
            "for every sample, dynamic target/history q05/q95 signed scaling (zero=50) "
            "and external-feature mean/std are fitted only on observations available "
            "through that sample's forecast origin; normalized external values and masks "
            f"are then scaled by {EXTERNAL_VALUE_SCALE:g} and {EXTERNAL_MASK_SCALE:g}, respectively"
        ),
        "topk_subsets_retrained_from_scratch": True,
        "topk_evaluation": "three-seed prediction ensemble; per-seed metrics retained separately",
        "masked_k_evaluation_used": False,
        "one_se_rule_used": True,
        "one_se_rule": "paired fold-wise difference versus raw-best K; choose smallest eligible K",
        "topk_fit_count": int(len(folds) * len(SEEDS) * len(topk_grid)),
        "run_signature": run_signature,
        "final_fi_metrics": final_fi_metrics,
        "raw_best_k": int(raw_best_row["k"]),
        "raw_best_cv_selection_score": float(raw_best_row["selection_score"]),
        "selected_k": selected_k,
        "zero_change_cv_mae": zero_component_mae,
        "zero_change_cv_ici_mae": zero_ici_mae,
        "zero_change_cv_selection_score": zero_selection_score,
        "history_only_cv_mae": float(history_row["mean_mae"]),
        "history_only_cv_ici_mae": float(history_row["mean_ici_mae"]),
        "selected_retrained_cv_mae": selected_component,
        "selected_retrained_cv_ici_mae": selected_ici,
        "selected_cv_selection_score": float(best_row["selection_score"]),
        "selected_beats_zero_change_baseline": bool(
            float(best_row["selection_score"]) < zero_selection_score
        ),
        "model_value_status": (
            "BEATS_ZERO_CHANGE_BASELINE"
            if float(best_row["selection_score"]) < zero_selection_score
            else "DOES_NOT_BEAT_ZERO_CHANGE_BASELINE"
        ),
        "full_retrained_cv_mae": float(full_row["mean_mae"]),
        "full_retrained_cv_ici_mae": float(full_row["mean_ici_mae"]),
        "selected_channel_count": selected_channel_count,
        "selected_model_input_columns": len(HISTORY) + selected_channel_count * 2,
        "validation_split_used_for_selection": False,
        "test_split_used_for_selection": False,
        "folds": fold_meta,
        "output_rows": split_rows,
        "outputs": {
            split: str((OUTPUT_DIR / f"selected_{split}.csv").relative_to(ROOT))
            for split in split_rows
        },
        "topk_detail_csv": str(TOPK_DETAIL_CSV.relative_to(ROOT)),
        "topk_curve_csv": str((RESULTS_DIR / "topk_cv_performance.csv").relative_to(ROOT)),
        "plots": plots,
        "status": "PASS",
    }
    selected_k, summary = finalize_topk_with_validation(
        gtnet,
        raw,
        registry,
        final_ranking,
        k_curve,
        topk_ensemble,
        device,
        zero_component_mae,
        zero_ici_mae,
        summary,
    )
    (RESULTS_DIR / "feature_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[6/6] PASS")
    print(json.dumps({
        "selected_k": selected_k,
        "selected_channel_count": int(summary["selected_channel_count"]),
        "history_only_cv_mae": float(history_row["mean_mae"]),
        "selected_cv_mae": float(summary["selected_retrained_cv_mae"]),
        "selected_cv_ici_mae": float(summary["selected_retrained_cv_ici_mae"]),
        "selected_validation_score": float(summary["selected_validation_selection_score"]),
        "zero_change_cv_mae": zero_component_mae,
        "zero_change_cv_ici_mae": zero_ici_mae,
        "selected_beats_zero_change_validation": bool(summary["selected_beats_zero_change_validation"]),
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
