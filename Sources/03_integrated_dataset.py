#!/usr/bin/env python3
"""Assemble PRISM's quarterly modeling dataset.

Rows are (quarter, industry node). Columns contain:
  - ex-post outcome targets (three ICI components + deterministic ICI),
  - causal/as-of competitiveness history available by that quarter's cutoff,
  - every node-specific feature channel from stage 02,
  - every global/context feature channel from stage 02.

Target construction is defined directly in this stage from the official source
panels collected by stage 01.
No feature selection, scaling, imputation, or model fitting is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
JOB = "03_integrated_dataset"
DATA_DIR = ROOT / "Data" / JOB
RESULTS_DIR = ROOT / "Results" / JOB
PLOTS_DIR = RESULTS_DIR / "plots"

TARGET_DATA_DIR = ROOT / "Data" / "01_target_data_collection" / "processed"
NODE_MAP = ROOT / "Results" / "01_target_data_collection" / "node_source_mapping.csv"

NODE_FEATURES = ROOT / "Data" / "02_initial_feature_pool" / "processed" / "node_features_quarterly_long.csv"
NODE_MASKS = ROOT / "Data" / "02_initial_feature_pool" / "processed" / "node_feature_masks_quarterly_long.csv"
GLOBAL_FEATURES = ROOT / "Data" / "02_initial_feature_pool" / "processed" / "global_features_quarterly.csv"
FEATURE_SUMMARY = ROOT / "Results" / "02_initial_feature_pool" / "feature_pool_summary.json"
FEATURE_REGISTRY = ROOT / "Results" / "02_initial_feature_pool" / "initial_feature_registry.csv"
TARGET_RELEASE_METADATA = TARGET_DATA_DIR / "target_release_metadata.csv"
FEATURE_RELEASE_METADATA = ROOT / "Data" / "02_initial_feature_pool" / "processed" / "feature_release_metadata.csv"
PIT_HISTORY_RAW = TARGET_DATA_DIR / "point_in_time_history_raw.csv"
PIT_HISTORY_AUDIT = ROOT / "Results" / "01_target_data_collection" / "point_in_time_history_audit.csv"

HISTORY_ANNUAL_CURRENT_VA = TARGET_DATA_DIR / "history_annual_current_value_added_56.csv"
HISTORY_ANNUAL_REAL_VA = TARGET_DATA_DIR / "history_annual_real_value_added_56.csv"
HISTORY_ANNUAL_CURRENT_GO = TARGET_DATA_DIR / "history_annual_current_gross_output_56.csv"
HISTORY_ANNUAL_REAL_GO = TARGET_DATA_DIR / "history_annual_real_gross_output_56.csv"
HISTORY_ANNUAL_GOS = TARGET_DATA_DIR / "history_annual_gos_56.csv"
HISTORY_ANNUAL_EMPLOYMENT = TARGET_DATA_DIR / "history_annual_employment_56.csv"

TRAIN_CSV = DATA_DIR / "raw_train.csv"
VAL_CSV = DATA_DIR / "raw_val.csv"
TEST_CSV = DATA_DIR / "raw_test.csv"
LEGACY_OUTPUTS = [
    DATA_DIR / "prism_integrated_quarterly.csv",
    DATA_DIR / "prism_integrated_feature_masks.csv",
    DATA_DIR / "raw_holdout_2025.csv",
]

TRAIN_END = "2019Q4"
VAL_START = "2020Q1"
VAL_END = "2022Q4"
TEST_START = "2023Q1"
TEST_END = "2025Q4"
MODEL_START = "2006Q1"

TARGET_COLUMNS = [
    "target_profitability",
    "target_productivity",
    "target_growth",
    "target_ici",
]
TARGET_RAW_COLUMNS = [
    "target_profitability_raw",
    "target_productivity_raw",
    "target_growth_raw",
]
HISTORY_COLUMNS = [
    "history_profitability",
    "history_productivity",
    "history_growth",
    "history_ici",
]
HISTORY_RAW_COLUMNS = [
    "history_profitability_raw",
    "history_productivity_raw",
    "history_growth_raw",
]

NETWORK_TOP_EDGES = 24
NETWORK_PLOT_EDGES = 16
EGO_EDGE_COUNT = 5


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def require_inputs() -> None:
    required = [
        NODE_MAP,
        NODE_FEATURES,
        NODE_MASKS,
        GLOBAL_FEATURES,
        FEATURE_SUMMARY,
        FEATURE_REGISTRY,
        TARGET_DATA_DIR / "current_value_added_56.csv",
        TARGET_DATA_DIR / "current_gross_output_56.csv",
        TARGET_DATA_DIR / "real_value_added_56.csv",
        TARGET_DATA_DIR / "real_gross_output_56.csv",
        TARGET_DATA_DIR / "annual_gos_56.csv",
        TARGET_DATA_DIR / "target_quarterly_employment_56.csv",
        TARGET_DATA_DIR / "bea_quarterly_56_long.csv",
        TARGET_RELEASE_METADATA,
        FEATURE_RELEASE_METADATA,
        PIT_HISTORY_RAW,
        PIT_HISTORY_AUDIT,
        HISTORY_ANNUAL_CURRENT_VA,
        HISTORY_ANNUAL_REAL_VA,
        HISTORY_ANNUAL_CURRENT_GO,
        HISTORY_ANNUAL_REAL_GO,
        HISTORY_ANNUAL_GOS,
        HISTORY_ANNUAL_EMPLOYMENT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisite files:\n" + "\n".join(missing))


def read_wide(path: Path, index_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={index_col: str})
    if index_col not in df.columns:
        raise RuntimeError(f"{path.name}: missing index column {index_col}")
    df = df.set_index(index_col)
    return df.apply(pd.to_numeric, errors="raise")


def quarter_end(quarter: str) -> pd.Timestamp:
    return pd.Period(quarter, freq="Q").end_time.normalize()


def proportional_denton(indicator: np.ndarray, annual: np.ndarray) -> np.ndarray:
    """First-difference proportional Denton used by the target pilot.

    Quarterly BEA flow values are SAAR, therefore the four quarterly values in
    each year must average to the annual benchmark.
    """
    t = indicator.size
    if t != annual.size * 4:
        raise ValueError(f"Denton length mismatch: quarters={t}, years={annual.size}")
    if np.any(np.abs(indicator) < 1e-12):
        raise ValueError("Denton indicator contains zero")

    d = np.zeros((t - 1, t), dtype=np.float64)
    for i in range(t - 1):
        d[i, i] = -1.0
        d[i, i + 1] = 1.0
    h = d.T @ d + np.eye(t) * 1e-10

    c = np.zeros((annual.size, t), dtype=np.float64)
    for y in range(annual.size):
        c[y, 4 * y : 4 * y + 4] = indicator[4 * y : 4 * y + 4]
    b = 4.0 * annual

    kkt = np.block([[h, c.T], [c, np.zeros((annual.size, annual.size))]])
    rhs = np.concatenate([np.zeros(t), b])
    ratio = np.linalg.solve(kkt, rhs)[:t]
    return indicator * ratio


def node_names() -> dict[str, str]:
    d = pd.read_csv(
        TARGET_DATA_DIR / "bea_quarterly_56_long.csv",
        usecols=["node_key", "node_name"],
        dtype=str,
    ).drop_duplicates("node_key")
    if len(d) != 56:
        raise RuntimeError(f"Expected 56 unique nodes, found {len(d)}")
    return dict(zip(d["node_key"], d["node_name"]))


def signed_robust_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return robust negative/positive scales around the economically neutral zero.

    Dynamic competitiveness components are centered conceptually at zero: no
    year-over-year change means neutral. The 5th/95th percentiles determine the
    two tails so one crisis/outlier cannot define the entire 0-100 score range.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("Cannot fit robust score bounds to an empty component")
    lo, hi = np.quantile(finite, [0.05, 0.95])
    if not (lo < 0.0 < hi):
        raise RuntimeError(f"Dynamic component does not straddle zero: q05={lo}, q95={hi}")
    return float(lo), float(hi)


def signed_robust_score(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map zero->50, q05->0 and q95->100, clipping only the extreme 5% tails."""
    arr = np.asarray(values, dtype=np.float64)
    score = np.where(
        arr >= 0.0,
        50.0 + 50.0 * arr / hi,
        50.0 + 50.0 * arr / abs(lo),
    )
    return np.clip(score, 0.0, 100.0)


def build_targets() -> tuple[pd.DataFrame, dict]:
    current_va = read_wide(TARGET_DATA_DIR / "current_value_added_56.csv", "quarter")
    current_go = read_wide(TARGET_DATA_DIR / "current_gross_output_56.csv", "quarter")
    real_va = read_wide(TARGET_DATA_DIR / "real_value_added_56.csv", "quarter")
    real_go = read_wide(TARGET_DATA_DIR / "real_gross_output_56.csv", "quarter")
    employment = read_wide(TARGET_DATA_DIR / "target_quarterly_employment_56.csv", "quarter")
    annual_gos = read_wide(TARGET_DATA_DIR / "annual_gos_56.csv", "year")

    # Stage 01 intentionally follows the latest official releases. The modeling
    # experiment uses the complete 2006Q1..2025Q4 panel with a 70/15/15
    # chronological split (56/12/12 quarters).
    source_start = str(pd.Period(MODEL_START, freq="Q") - 4)
    quarterly_index = pd.period_range(source_start, TEST_END, freq="Q").astype(str).tolist()
    annual_index = [str(y) for y in range(int(source_start[:4]), int(TEST_END[:4]) + 1)]
    quarterly_frames = [current_va, current_go, real_va, real_go, employment]
    for frame in quarterly_frames:
        missing = [q for q in quarterly_index if q not in frame.index]
        if missing:
            raise RuntimeError(f"Quarterly target source missing frozen experiment periods: {missing[:8]}")
    missing_annual = [y for y in annual_index if y not in annual_gos.index]
    if missing_annual:
        raise RuntimeError(f"Annual GOS missing frozen experiment years: {missing_annual}")
    current_va = current_va.loc[quarterly_index]
    current_go = current_go.loc[quarterly_index]
    real_va = real_va.loc[quarterly_index]
    real_go = real_go.loc[quarterly_index]
    employment = employment.loc[quarterly_index]
    annual_gos = annual_gos.loc[annual_index]

    frames = [current_go, real_va, real_go, employment]
    if not all(current_va.index.equals(x.index) and current_va.columns.equals(x.columns) for x in frames):
        raise RuntimeError("Quarterly target source panels are not aligned")
    expected_quarters = len(quarterly_index)
    expected_years = len(annual_index)
    if current_va.shape != (expected_quarters, 56):
        raise RuntimeError(f"Unexpected quarterly target shape: {current_va.shape}")
    if annual_gos.shape != (expected_years, 56):
        raise RuntimeError(f"Unexpected annual GOS shape: {annual_gos.shape}")

    gos_q = np.empty_like(current_va.to_numpy(dtype=np.float64))
    max_benchmark_error = 0.0
    for j in range(current_va.shape[1]):
        gos_q[:, j] = proportional_denton(
            current_va.iloc[:, j].to_numpy(dtype=np.float64),
            annual_gos.iloc[:, j].to_numpy(dtype=np.float64),
        )
        err = np.max(
            np.abs(
                gos_q[:, j].reshape(expected_years, 4).mean(axis=1)
                - annual_gos.iloc[:, j].to_numpy(dtype=np.float64)
            )
        )
        max_benchmark_error = max(max_benchmark_error, float(err))

    nodes = current_va.columns.tolist()
    profitability_level = gos_q / current_go.to_numpy(dtype=np.float64)

    # Dynamic target definition: every component measures one-year competitive
    # improvement/deterioration rather than a mostly static industry level.
    # Real-VA ratios are invariant to BEA chained-dollar rebasing.
    rva = real_va.to_numpy(dtype=np.float64).copy()
    emp = employment.to_numpy(dtype=np.float64)
    funds_idx = nodes.index("funds_trusts")
    securities_idx = nodes.index("securities")
    # Match the continuous CES 523+525 employment proxy used for funds/trusts.
    rva[:, funds_idx] = rva[:, funds_idx] + rva[:, securities_idx]

    profitability_raw = np.full_like(profitability_level, np.nan)
    productivity_raw = np.full_like(rva, np.nan)
    growth_raw = np.full_like(real_go.to_numpy(dtype=np.float64), np.nan)
    profitability_raw[4:] = profitability_level[4:] - profitability_level[:-4]
    productivity_now = rva[4:] / emp[4:]
    productivity_prev = rva[:-4] / emp[:-4]
    productivity_scale = np.abs(productivity_now) + np.abs(productivity_prev)
    if np.any(productivity_scale <= 1e-15):
        raise RuntimeError("Degenerate labor-productivity symmetric-change denominator")
    productivity_raw[4:] = (
        2.0 * (productivity_now - productivity_prev) / productivity_scale
    )
    real_go_arr = real_go.to_numpy(dtype=np.float64)
    growth_raw[4:] = real_go_arr[4:] / real_go_arr[:-4] - 1.0

    raw = {
        "profitability": profitability_raw[4:],
        "productivity": productivity_raw[4:],
        "growth": growth_raw[4:],
    }
    quarters = current_va.index[4:].tolist()

    # Robust signed scoring: q05 -> 0, zero change -> 50, q95 -> 100.
    # Bounds are fitted only on Train and are descriptive; stage 05 refits the
    # same transform expanding through each forecast origin.
    train_mask = np.asarray([MODEL_START <= q <= TRAIN_END for q in quarters])
    bounds: dict[str, tuple[float, float]] = {}
    for name, arr in raw.items():
        train_vals = arr[train_mask]
        bounds[name] = signed_robust_bounds(train_vals)

    score = {}
    for name, arr in raw.items():
        lo, hi = bounds[name]
        score[name] = signed_robust_score(arr, lo, hi)
    ici = (score["profitability"] + score["productivity"] + score["growth"]) / 3.0

    extrema = []
    for component, arr in raw.items():
        train_arr = arr[train_mask]
        train_quarters = np.asarray(quarters)[train_mask]
        min_flat = int(np.nanargmin(train_arr))
        max_flat = int(np.nanargmax(train_arr))
        min_t, min_n = np.unravel_index(min_flat, train_arr.shape)
        max_t, max_n = np.unravel_index(max_flat, train_arr.shape)
        extrema.append({
            "component": component,
            "train_raw_min": float(train_arr[min_t, min_n]),
            "train_raw_min_quarter": str(train_quarters[min_t]),
            "train_raw_min_node": nodes[min_n],
            "train_raw_max": float(train_arr[max_t, max_n]),
            "train_raw_max_quarter": str(train_quarters[max_t]),
            "train_raw_max_node": nodes[max_n],
        })

    rows = []
    names = node_names()
    for t, quarter in enumerate(quarters):
        for j, node in enumerate(nodes):
            rows.append({
                "quarter": quarter,
                "node_key": node,
                "node_name": names[node],
                "target_profitability_raw": float(raw["profitability"][t, j]),
                "target_productivity_raw": float(raw["productivity"][t, j]),
                "target_growth_raw": float(raw["growth"][t, j]),
                "target_profitability": float(score["profitability"][t, j]),
                "target_productivity": float(score["productivity"][t, j]),
                "target_growth": float(score["growth"][t, j]),
                "target_ici": float(ici[t, j]),
            })
    target = pd.DataFrame(rows)
    meta = {
        "quarters": [quarters[0], quarters[-1]],
        "quarter_count": len(quarters),
        "node_count": len(nodes),
        "rows": len(target),
        "normalization_bounds": bounds,
        "normalization_extrema": extrema,
        "denton_max_annual_benchmark_abs_error": max_benchmark_error,
    }
    return target, meta


def build_causal_history(target_meta: dict) -> tuple[pd.DataFrame, dict]:
    """Load point-in-time dynamic history and add expanding robust scores.

    Raw values are authoritative for modeling. Diagnostic scores use only
    observations known through quarter t and preserve the economically neutral
    zero at score 50.
    """
    del target_meta  # static target bounds are intentionally not used for history.
    history = pd.read_csv(PIT_HISTORY_RAW, dtype={"quarter": str, "node_key": str, "node_name": str})
    audit = pd.read_csv(PIT_HISTORY_AUDIT, dtype=str)
    history = history[(history["quarter"] >= MODEL_START) & (history["quarter"] <= TEST_END)].copy()
    audit = audit[(audit["quarter"] >= MODEL_START) & (audit["quarter"] <= TEST_END)].copy()
    expected_rows = len(pd.period_range(MODEL_START, TEST_END, freq="Q")) * 56
    if len(history) != expected_rows or history.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError("Invalid stage-01 point-in-time history panel")
    if history[HISTORY_RAW_COLUMNS].isna().any().any():
        raise RuntimeError("Point-in-time raw history is incomplete")
    if not np.isfinite(history[HISTORY_RAW_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("Point-in-time raw history contains non-finite values")

    scored = []
    for quarter in history["quarter"].drop_duplicates().tolist():
        observed = history[history["quarter"] <= quarter]
        current = history[history["quarter"] == quarter].copy()
        component_scores = []
        for raw_col, score_col in zip(HISTORY_RAW_COLUMNS, HISTORY_COLUMNS[:3]):
            lo, hi = signed_robust_bounds(observed[raw_col].to_numpy(dtype=float))
            current[score_col] = signed_robust_score(
                current[raw_col].to_numpy(dtype=float), lo, hi
            )
            component_scores.append(current[score_col])
        current["history_ici"] = sum(component_scores) / 3.0
        scored.append(current)
    history = pd.concat(scored, ignore_index=True)
    # node_name is authoritative on the target panel; avoid merge suffixes.
    history = history.drop(columns=["node_name"], errors="ignore")

    cutoff = pd.to_datetime(audit["information_cutoff_date"])
    bea = pd.to_datetime(audit["bea_vintage_release_date"])
    emp = pd.to_datetime(audit["employment_available_date"])
    if ((bea > cutoff) | (emp > cutoff)).any():
        raise RuntimeError("Point-in-time source availability exceeds forecast cutoff")
    audit.to_csv(RESULTS_DIR / "information_cutoff_audit.csv", index=False, encoding="utf-8-sig")
    component_resolution_rows = {}
    component_lag_summary = {}
    origin_period = audit["quarter"].map(lambda x: pd.Period(x, freq="Q"))
    for component in ("profitability", "productivity", "growth"):
        res_col = f"{component}_resolution"
        src_col = f"{component}_source_period"
        if res_col not in audit.columns or src_col not in audit.columns:
            continue
        component_resolution_rows[component] = {
            "annual": int((audit[res_col] == "ANNUAL").sum()),
            "quarterly": int((audit[res_col] == "QUARTERLY").sum()),
        }
        lags = []
        for origin, resolution, source in zip(origin_period, audit[res_col], audit[src_col]):
            source_period = (
                pd.Period(str(source), freq="Q")
                if resolution == "QUARTERLY"
                else pd.Period(f"{source}Q4", freq="Q")
            )
            lags.append((origin - source_period).n)
        lag_arr = np.asarray(lags, dtype=float)
        component_lag_summary[component] = {
            "mean_quarters": float(lag_arr.mean()),
            "median_quarters": float(np.median(lag_arr)),
            "p90_quarters": float(np.quantile(lag_arr, 0.90)),
            "max_quarters": float(lag_arr.max()),
        }

    meta = {
        "period": [MODEL_START, TEST_END],
        "rows": len(history),
        "available_rows": len(history),
        "missing_rows": 0,
        "annual_rows": int((audit["state_resolution"] == "ANNUAL").sum()),
        "quarterly_rows": int((audit["state_resolution"] == "QUARTERLY").sum()),
        "component_resolution_rows": component_resolution_rows,
        "component_source_lag": component_lag_summary,
        "bea_vintages_used": int(audit["bea_vintage_release_date"].nunique()),
        "parent_gos_proxy_rows": int(audit["gos_mode"].str.startswith("PARENT_SHARE:").sum()),
        "policy": (
            "official point-in-time BEA vintage + finalized/vintage employment; "
            "each component uses its freshest independently available source period; "
            "expanding normalization through each origin"
        ),
    }
    (RESULTS_DIR / "information_cutoff_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return history, meta


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    node = pd.read_csv(NODE_FEATURES, dtype={"quarter": str, "node_key": str})
    masks = pd.read_csv(NODE_MASKS, dtype={"quarter": str, "node_key": str})
    global_df = pd.read_csv(GLOBAL_FEATURES, dtype={"quarter": str})

    node_feature_cols = [c for c in node.columns if c not in {"quarter", "node_key"}]
    mask_feature_cols = [c for c in masks.columns if c not in {"quarter", "node_key"}]
    global_feature_cols = [c for c in global_df.columns if c != "quarter"]

    if node_feature_cols != mask_feature_cols:
        raise RuntimeError("Node feature and mask columns differ")
    feature_summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    expected_node = int(feature_summary["node_model_channels_pass"])
    expected_global = int(feature_summary["global_model_channels_pass"])
    if len(node_feature_cols) != expected_node:
        raise RuntimeError(f"Expected {expected_node} node feature channels, found {len(node_feature_cols)}")
    if len(global_feature_cols) != expected_global:
        raise RuntimeError(f"Expected {expected_global} global feature channels, found {len(global_feature_cols)}")
    if set(node_feature_cols) & set(global_feature_cols):
        raise RuntimeError("Node/global feature names overlap")

    return node, masks, node_feature_cols, global_feature_cols


def save_target_diagnostics(
    integrated: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_meta: dict,
) -> list[str]:
    """Write descriptive target tables and plots without fitting any model."""
    generated: list[str] = []
    # Avoid Windows overwrite/locking edge cases from a prior run.
    for old_plot in PLOTS_DIR.glob("*.png"):
        try:
            old_plot.unlink()
        except OSError:
            pass
    label = {
        "target_profitability": "Profitability",
        "target_productivity": "Productivity",
        "target_growth": "Growth",
        "target_ici": "ICI",
    }
    splits = {"Train": train, "Validation": val, "Test": test}

    # Exact raw-component normalization bounds used to construct the scores.
    extrema_by_component = {x["component"]: x for x in target_meta["normalization_extrema"]}
    bounds_rows = []
    for component, (lo, hi) in target_meta["normalization_bounds"].items():
        extreme = extrema_by_component[component]
        bounds_rows.append({
            "component": component,
            "train_raw_min": extreme["train_raw_min"],
            "train_raw_min_quarter": extreme["train_raw_min_quarter"],
            "train_raw_min_node": extreme["train_raw_min_node"],
            "train_raw_max": extreme["train_raw_max"],
            "train_raw_max_quarter": extreme["train_raw_max_quarter"],
            "train_raw_max_node": extreme["train_raw_max_node"],
            "train_raw_q05": lo,
            "train_raw_q95": hi,
            "raw_range": extreme["train_raw_max"] - extreme["train_raw_min"],
            "formula": "zero->50; q05->0; q95->100; piecewise-linear signed scaling",
            "clipped": True,
        })
    pd.DataFrame(bounds_rows).to_csv(
        RESULTS_DIR / "target_normalization_bounds.csv", index=False, encoding="utf-8-sig"
    )

    # Split-wise distribution summary.
    summary_rows = []
    for split_name, df in splits.items():
        for col in TARGET_COLUMNS:
            s = pd.to_numeric(df[col], errors="raise")
            summary_rows.append({
                "split": split_name,
                "target": col,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
                "below_0": int((s < 0).sum()),
                "above_100": int((s > 100).sum()),
            })
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "target_distribution_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Overall and split correlations.
    corr_all = integrated[TARGET_COLUMNS].corr(method="pearson")
    corr_all.to_csv(RESULTS_DIR / "target_correlations_all.csv", encoding="utf-8-sig")
    for split_name, df in splits.items():
        df[TARGET_COLUMNS].corr(method="pearson").to_csv(
            RESULTS_DIR / f"target_correlations_{split_name.lower()}.csv", encoding="utf-8-sig"
        )

    # Industry-level target summaries.
    industry_summary = (
        integrated.groupby(["node_key", "node_name"])[TARGET_COLUMNS]
        .agg(["mean", "std", "min", "max"])
    )
    industry_summary.columns = [f"{a}_{b}" for a, b in industry_summary.columns]
    industry_summary.reset_index().to_csv(
        RESULTS_DIR / "target_industry_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 1) Train / validation / test distributions.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), TARGET_COLUMNS):
        values = integrated[col].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        bins = np.linspace(lo, hi, 36)
        for split_name, df in splits.items():
            ax.hist(df[col], bins=bins, density=True, histtype="step", linewidth=1.8, label=split_name)
        ax.set_title(label[col])
        ax.set_xlabel("Normalized score")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Target distributions by chronological split")
    fig.tight_layout()
    path = PLOTS_DIR / "01_target_distributions_by_split.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 2) Target correlation heatmap.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    mat = corr_all.to_numpy(dtype=float)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(TARGET_COLUMNS)), [label[c] for c in TARGET_COLUMNS], rotation=30, ha="right")
    ax.set_yticks(range(len(TARGET_COLUMNS)), [label[c] for c in TARGET_COLUMNS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set_title("Target correlation structure")
    fig.tight_layout()
    path = PLOTS_DIR / "02_target_correlation_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # Prepare quarterly cross-sectional panels.
    quarter_order = integrated["quarter"].drop_duplicates().tolist()
    x = np.arange(len(quarter_order))
    qgroups = integrated.groupby("quarter", sort=False)

    # 3) Cross-sectional median and IQR over time.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, col in zip(axes.ravel(), TARGET_COLUMNS):
        med = qgroups[col].median().reindex(quarter_order).to_numpy()
        q25 = qgroups[col].quantile(0.25).reindex(quarter_order).to_numpy()
        q75 = qgroups[col].quantile(0.75).reindex(quarter_order).to_numpy()
        ax.plot(x, med, linewidth=1.8, label="Median")
        ax.fill_between(x, q25, q75, alpha=0.22, label="25–75% across industries")
        ax.set_title(label[col])
        ax.grid(alpha=0.25)
    for ax in axes.ravel():
        ax.axvline(quarter_order.index(VAL_START), linestyle="--", linewidth=1)
        ax.axvline(quarter_order.index(TEST_START), linestyle="--", linewidth=1)
    tick_idx = np.arange(0, len(quarter_order), 8)
    for ax in axes[-1]:
        ax.set_xticks(tick_idx, [quarter_order[i] for i in tick_idx], rotation=45, ha="right")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Cross-industry target dynamics over time")
    fig.tight_layout()
    path = PLOTS_DIR / "03_target_time_dynamics_median_iqr.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 4) Cross-sectional dispersion: standard deviation across 56 industries.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col in TARGET_COLUMNS:
        sd = qgroups[col].std(ddof=1).reindex(quarter_order).to_numpy()
        ax.plot(x, sd, linewidth=1.6, label=label[col])
    ax.axvline(quarter_order.index(VAL_START), linestyle="--", linewidth=1)
    ax.axvline(quarter_order.index(TEST_START), linestyle="--", linewidth=1)
    ax.set_xticks(tick_idx, [quarter_order[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_ylabel("Cross-industry standard deviation")
    ax.set_title("Cross-sectional dispersion of competitiveness dimensions")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = PLOTS_DIR / "04_target_cross_industry_dispersion.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 5) Component score vs deterministic ICI.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, TARGET_COLUMNS[:3]):
        hb = ax.hexbin(integrated[col], integrated["target_ici"], gridsize=38, mincnt=1, cmap="viridis")
        r = float(integrated[[col, "target_ici"]].corr().iloc[0, 1])
        ax.set_xlabel(label[col])
        ax.set_ylabel("ICI")
        ax.set_title(f"{label[col]} vs ICI  (r={r:.2f})")
        ax.grid(alpha=0.15)
        fig.colorbar(hb, ax=ax, label="Observations")
    fig.suptitle("Component–ICI relationship across industry-quarter observations")
    fig.tight_layout()
    path = PLOTS_DIR / "05_component_vs_ici_relationship.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 6) Persistence / one-quarter transition dynamics.
    ordered = integrated.copy()
    ordered["quarter_index"] = ordered["quarter"].map({q: i for i, q in enumerate(quarter_order)})
    ordered = ordered.sort_values(["node_key", "quarter_index"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    transition_rows = []
    for ax, col in zip(axes.ravel(), TARGET_COLUMNS):
        x0 = ordered.groupby("node_key")[col].shift(1)
        y1 = ordered[col]
        mask = x0.notna() & y1.notna()
        xv = x0[mask].to_numpy(dtype=float)
        yv = y1[mask].to_numpy(dtype=float)
        corr = float(np.corrcoef(xv, yv)[0, 1])
        transition_rows.append({"target": col, "lag1_correlation": corr, "n": int(mask.sum())})
        ax.scatter(xv, yv, s=7, alpha=0.25)
        mn = min(float(np.min(xv)), float(np.min(yv)))
        mx = max(float(np.max(xv)), float(np.max(yv)))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        ax.set_xlabel(f"{label[col]} at t-1")
        ax.set_ylabel(f"{label[col]} at t")
        ax.set_title(f"{label[col]} persistence  (r={corr:.3f})")
        ax.grid(alpha=0.2)
    pd.DataFrame(transition_rows).to_csv(
        RESULTS_DIR / "target_lag1_persistence.csv", index=False, encoding="utf-8-sig"
    )
    fig.suptitle("One-quarter target transition dynamics")
    fig.tight_layout()
    path = PLOTS_DIR / "06_target_lag1_transition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 7) Directed inter-industry lead-lag network based on ΔICI.
    # For each i -> j pair, estimate standardized OLS:
    #   ΔICI_j,t ~ ΔICI_j,t-1 + mean(ΔICI_*,t-1) + ΔICI_i,t-1
    # The final coefficient is a descriptive directional predictive association,
    # not a causal effect.
    ici_panel = integrated.pivot(index="quarter", columns="node_key", values="target_ici").reindex(quarter_order)
    network_quarters = train["quarter"].drop_duplicates().tolist()
    network_panel = train.pivot(index="quarter", columns="node_key", values="target_ici").reindex(network_quarters)
    node_name_map = (
        integrated[["node_key", "node_name"]]
        .drop_duplicates("node_key")
        .set_index("node_key")["node_name"]
        .to_dict()
    )
    dici = network_panel.diff()
    common_lag = dici.mean(axis=1).shift(1)
    edge_rows = []
    nodes_order = list(network_panel.columns)

    def zscore(a: np.ndarray) -> np.ndarray:
        sd = float(np.std(a, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.full_like(a, np.nan, dtype=float)
        return (a - float(np.mean(a))) / sd

    for source in nodes_order:
        source_lag = dici[source].shift(1)
        for target_node in nodes_order:
            if source == target_node:
                continue
            y = dici[target_node]
            own_lag = dici[target_node].shift(1)
            frame = pd.concat(
                [y.rename("y"), source_lag.rename("x"), own_lag.rename("own"), common_lag.rename("common")],
                axis=1,
            ).dropna()
            if len(frame) < 20:
                continue
            yz = zscore(frame["y"].to_numpy(dtype=float))
            xz = zscore(frame["x"].to_numpy(dtype=float))
            ownz = zscore(frame["own"].to_numpy(dtype=float))
            commonz = zscore(frame["common"].to_numpy(dtype=float))
            if not all(np.isfinite(v).all() for v in [yz, xz, ownz, commonz]):
                continue
            X = np.column_stack([np.ones(len(frame)), ownz, commonz, xz])
            coef, _, _, _ = np.linalg.lstsq(X, yz, rcond=None)
            beta = float(coef[-1])
            lead_corr = float(np.corrcoef(xz, yz)[0, 1])
            edge_rows.append({
                "source": source,
                "source_name": node_name_map[source],
                "target": target_node,
                "target_name": node_name_map[target_node],
                "standardized_beta": beta,
                "lead_lag_corr": lead_corr,
                "abs_beta": abs(beta),
                "sign": "positive" if beta >= 0 else "negative",
                "n": int(len(frame)),
                "estimation_period": f"{network_quarters[0]}-{network_quarters[-1]}",
                "definition": "dICI_source(t-1) -> dICI_target(t), controlling target own lag and lagged common dICI",
            })

    edges = pd.DataFrame(edge_rows).sort_values("abs_beta", ascending=False).reset_index(drop=True)
    edges.to_csv(RESULTS_DIR / "interindustry_dynamics_edges.csv", index=False, encoding="utf-8-sig")

    # Keep one dominant direction per unordered pair, then the strongest edges only.
    pair_best = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in edges.itertuples(index=False):
        pair = tuple(sorted((row.source, row.target)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        pair_best.append(row._asdict())
    top_edges = pd.DataFrame(pair_best).head(NETWORK_TOP_EDGES).copy()
    top_edges.to_csv(RESULTS_DIR / "interindustry_dynamics_top_edges.csv", index=False, encoding="utf-8-sig")

    graph = nx.DiGraph()
    for row in top_edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.abs_beta, beta=row.standardized_beta)

    node_strength = {}
    for node in graph.nodes:
        incoming = sum(abs(d["beta"]) for _, _, d in graph.in_edges(node, data=True))
        outgoing = sum(abs(d["beta"]) for _, _, d in graph.out_edges(node, data=True))
        node_strength[node] = incoming + outgoing

    node_summary_rows = []
    for node in nodes_order:
        in_all = edges[edges["target"] == node]
        out_all = edges[edges["source"] == node]
        node_summary_rows.append({
            "node_key": node,
            "node_name": node_name_map[node],
            "mean_ici": float(ici_panel[node].mean()),
            "recent_ici": float(ici_panel[node].iloc[-1]),
            "incoming_abs_beta_sum": float(in_all["abs_beta"].sum()),
            "outgoing_abs_beta_sum": float(out_all["abs_beta"].sum()),
            "selected_network_strength": float(node_strength.get(node, 0.0)),
        })
    node_summary = pd.DataFrame(node_summary_rows).sort_values("selected_network_strength", ascending=False)
    node_summary.to_csv(RESULTS_DIR / "interindustry_dynamics_node_summary.csv", index=False, encoding="utf-8-sig")

    # Figure-2-style network: same visual grammar as the cyber-threat figure.
    # Node size = selected-network strength; edge distance/opacity = |beta|;
    # edge color = sign; arrowhead = lead -> lag direction.
    plot_edges = top_edges.head(NETWORK_PLOT_EDGES).copy()
    plot_graph = nx.DiGraph()
    for row in plot_edges.itertuples(index=False):
        plot_graph.add_edge(
            row.source,
            row.target,
            beta=row.standardized_beta,
            strength=row.abs_beta,
        )

    industry_code = {
        "plastics_rubber": "PR",
        "textiles": "TX",
        "wholesale": "WH",
        "rail_transport": "RA",
        "truck_transport": "TRK",
        "furniture": "FU",
        "primary_metals": "PM",
        "fabricated_metal": "FM",
        "rental_leasing": "RL",
        "amusement_recreation": "AR",
        "accommodation": "AC",
        "wood_products": "WD",
        "electrical_equipment": "EE",
        "retail": "RT",
        "misc_professional": "PS",
        "petroleum_coal": "PC",
        "oil_gas_extraction": "OG",
        "pipeline": "PL",
        "computer_electronic": "CE",
        "construction": "CN",
        "motor_vehicles": "MV",
    }
    pd.DataFrame([
        {
            "code": industry_code.get(node, node[:4].upper()),
            "node_key": node,
            "node_name": node_name_map[node],
        }
        for node in sorted(plot_graph.nodes())
    ]).to_csv(
        RESULTS_DIR / "interindustry_dynamics_node_codes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nodes_plot = sorted(plot_graph.nodes())
    selected_strength = {
        node: sum(d["strength"] for _, _, d in plot_graph.in_edges(node, data=True))
        + sum(d["strength"] for _, _, d in plot_graph.out_edges(node, data=True))
        for node in nodes_plot
    }
    max_strength = max(selected_strength.values())
    min_node_size = 260.0
    max_node_size = 1150.0
    node_sizes_plot = {
        node: min_node_size
        + (max_node_size - min_node_size)
        * (selected_strength[node] / max_strength) ** 0.50
        for node in nodes_plot
    }

    abs_beta = plot_edges["abs_beta"].to_numpy(dtype=float)
    beta_min, beta_max = float(abs_beta.min()), float(abs_beta.max())
    if beta_max <= beta_min:
        target_distance = {(r.source, r.target): 0.72 for r in plot_edges.itertuples(index=False)}
    else:
        target_distance = {
            (r.source, r.target): 0.34
            + 0.95 * (1.0 - (r.abs_beta - beta_min) / (beta_max - beta_min)) ** 0.75
            for r in plot_edges.itertuples(index=False)
        }

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(nodes_plot)
    for row in plot_edges.itertuples(index=False):
        normalized = 1.0 if beta_max <= beta_min else (row.abs_beta - beta_min) / (beta_max - beta_min)
        layout_graph.add_edge(row.source, row.target, weight=1.0 + 4.0 * normalized)
    node_index = {node: i for i, node in enumerate(nodes_plot)}
    spring_pos = nx.spring_layout(
        layout_graph,
        seed=4,
        weight=None,
        k=0.42,
        iterations=1200,
        threshold=1e-7,
        scale=1.0,
    )
    initial = np.asarray([spring_pos[node] for node in nodes_plot], dtype=float)

    def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-15:
            return float(np.linalg.norm(point - start))
        tt = float(np.dot(point - start, segment) / denominator)
        tt = min(1.0, max(0.0, tt))
        return float(np.linalg.norm(point - (start + tt * segment)))

    directed_edges = [(r.source, r.target) for r in plot_edges.itertuples(index=False)]

    def layout_residuals(flat: np.ndarray) -> np.ndarray:
        points = flat.reshape((-1, 2))
        values: list[float] = []

        # As in the cyber figure, stronger relationships are pulled closer while
        # retaining enough space for the large hub nodes.
        for first, second in directed_edges:
            p0 = points[node_index[first]]
            p1 = points[node_index[second]]
            distance = float(np.linalg.norm(p0 - p1))
            values.append((distance - target_distance[(first, second)]) * 88.0)

        # Node collision + extra visual spacing for non-neighbours.
        for first_index in range(len(nodes_plot)):
            for second_index in range(first_index + 1, len(nodes_plot)):
                first = nodes_plot[first_index]
                second = nodes_plot[second_index]
                distance = float(np.linalg.norm(points[first_index] - points[second_index]))
                required = 0.0028 * (
                    math.sqrt(node_sizes_plot[first]) + math.sqrt(node_sizes_plot[second])
                )
                values.append(max(0.0, required - distance) * 72.0)
                if not layout_graph.has_edge(first, second):
                    values.append(max(0.0, required * 3.0 - distance) * 3.0)

        # Keep edges out of unrelated nodes.
        for first, second in directed_edges:
            start = points[node_index[first]]
            end = points[node_index[second]]
            for node in nodes_plot:
                if node in (first, second):
                    continue
                required = 0.0028 * math.sqrt(node_sizes_plot[node]) * 1.22
                distance = point_segment_distance(points[node_index[node]], start, end)
                values.append(max(0.0, required - distance) * 82.0)

        # Spread high-degree hub neighbours angularly rather than stacking them.
        for hub in nodes_plot:
            neighbors = list(layout_graph.neighbors(hub))
            if len(neighbors) < 4:
                continue
            hub_point = points[node_index[hub]]
            ideal_angle = 2.0 * math.pi / len(neighbors)
            for i, first in enumerate(neighbors):
                first_vec = points[node_index[first]] - hub_point
                first_len = float(np.linalg.norm(first_vec))
                if first_len <= 1e-12:
                    continue
                for second in neighbors[i + 1:]:
                    second_vec = points[node_index[second]] - hub_point
                    second_len = float(np.linalg.norm(second_vec))
                    if second_len <= 1e-12:
                        continue
                    desired = math.sqrt(max(
                        0.0,
                        first_len ** 2 + second_len ** 2
                        - 2.0 * first_len * second_len * math.cos(ideal_angle),
                    ))
                    actual = float(np.linalg.norm(points[node_index[first]] - points[node_index[second]]))
                    values.append(max(0.0, desired - actual) * 7.5)

            directions = []
            for neighbor in neighbors:
                vector = points[node_index[neighbor]] - hub_point
                length = float(np.linalg.norm(vector))
                if length > 1e-12:
                    directions.append(vector / length)
            if directions:
                imbalance = np.mean(directions, axis=0)
                values.extend((imbalance * 3.4).tolist())

        # Keep the layout centered.
        values.extend((points.mean(axis=0) * 0.12).tolist())
        return np.asarray(values, dtype=float)

    fit = least_squares(
        layout_residuals,
        initial.ravel(),
        max_nfev=800,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    optimized = fit.x.reshape((-1, 2))

    # Rotate the two largest hubs onto the landscape axis, exactly as in the cyber figure.
    optimized -= optimized.mean(axis=0)
    hubs = sorted(nodes_plot, key=lambda n: (-layout_graph.degree[n], n))[:2]
    h0 = optimized[node_index[hubs[0]]]
    h1 = optimized[node_index[hubs[1]]]
    angle = -math.atan2((h1 - h0)[1], (h1 - h0)[0])
    c, s = math.cos(angle), math.sin(angle)
    rotated = np.column_stack([
        optimized[:, 0] * c - optimized[:, 1] * s,
        optimized[:, 0] * s + optimized[:, 1] * c,
    ])
    if rotated[node_index[hubs[0]], 0] > rotated[node_index[hubs[1]], 0]:
        rotated[:, 0] *= -1.0
    # Map the optimized layout into a fixed plotting box, then enforce overlap
    # constraints in physical point coordinates. Unlike a soft layout penalty,
    # this guarantees that rendered node circles do not overlap.
    layout_x = rotated[:, 0]
    layout_y = rotated[:, 1]
    x_span = max(float(layout_x.max() - layout_x.min()), 1e-9)
    y_span = max(float(layout_y.max() - layout_y.min()), 1e-9)
    normalized_pos = np.column_stack([
        0.06 + 0.88 * (layout_x - layout_x.min()) / x_span,
        0.13 + 0.72 * (layout_y - layout_y.min()) / y_span,
    ])

    fig, ax = plt.subplots(figsize=(14.8, 7.0), facecolor="white")
    # Reserve the right third of the canvas exclusively for the code legend.
    fig.subplots_adjust(left=0.02, right=0.64, top=0.875, bottom=0.075)
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    axes_box = ax.get_position()
    axes_width_pt = fig.get_figwidth() * 72.0 * axes_box.width
    axes_height_pt = fig.get_figheight() * 72.0 * axes_box.height
    physical = np.column_stack([
        normalized_pos[:, 0] * axes_width_pt,
        normalized_pos[:, 1] * axes_height_pt,
    ])
    node_radius_pt = np.asarray([
        math.sqrt(node_sizes_plot[node] / math.pi)
        for node in nodes_plot
    ], dtype=float)

    def clamp_physical(points: np.ndarray) -> None:
        border = 7.0
        for idx in range(len(nodes_plot)):
            radius = node_radius_pt[idx]
            points[idx, 0] = min(max(points[idx, 0], radius + border), axes_width_pt - radius - border)
            points[idx, 1] = min(max(points[idx, 1], radius + border), axes_height_pt - radius - border)

    def physical_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, np.ndarray]:
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-15:
            return float(np.linalg.norm(point - start)), start
        tt = float(np.dot(point - start, segment) / denominator)
        tt = min(1.0, max(0.0, tt))
        closest = start + tt * segment
        return float(np.linalg.norm(point - closest)), closest

    clamp_physical(physical)
    pair_padding_pt = 11.0
    edge_padding_pt = 6.0
    for _ in range(1800):
        changed = False

        # Hard node-node separation.
        for i in range(len(nodes_plot)):
            for j in range(i + 1, len(nodes_plot)):
                vector = physical[j] - physical[i]
                distance = float(np.linalg.norm(vector))
                required = node_radius_pt[i] + node_radius_pt[j] + pair_padding_pt
                if distance + 1e-9 >= required:
                    continue
                if distance <= 1e-9:
                    angle0 = ((i + 1) * 47 + (j + 1) * 31) * math.pi / 180.0
                    direction = np.asarray([math.cos(angle0), math.sin(angle0)])
                else:
                    direction = vector / distance
                push = 0.505 * (required - max(distance, 1e-9)) * direction
                physical[i] -= push
                physical[j] += push
                changed = True

        # Keep every unrelated node away from every rendered edge.
        for source, target in directed_edges:
            i = node_index[source]
            j = node_index[target]
            start = physical[i]
            end = physical[j]
            for node in nodes_plot:
                if node in (source, target):
                    continue
                k = node_index[node]
                distance, closest = physical_segment_distance(physical[k], start, end)
                required = node_radius_pt[k] + edge_padding_pt
                if distance + 1e-9 >= required:
                    continue
                vector = physical[k] - closest
                if float(np.linalg.norm(vector)) <= 1e-9:
                    edge_vector = end - start
                    vector = np.asarray([-edge_vector[1], edge_vector[0]], dtype=float)
                direction = vector / max(float(np.linalg.norm(vector)), 1e-9)
                physical[k] += (required - distance + 0.5) * direction
                changed = True

        clamp_physical(physical)
        if not changed:
            break

    # Machine-check the exact rendered circle clearances before saving.
    node_overlap_count = 0
    min_node_clearance_pt = math.inf
    for i in range(len(nodes_plot)):
        for j in range(i + 1, len(nodes_plot)):
            distance = float(np.linalg.norm(physical[j] - physical[i]))
            clearance = distance - node_radius_pt[i] - node_radius_pt[j]
            min_node_clearance_pt = min(min_node_clearance_pt, clearance)
            if clearance < -1e-6:
                node_overlap_count += 1

    edge_node_intrusions = 0
    min_edge_node_clearance_pt = math.inf
    for source, target in directed_edges:
        start = physical[node_index[source]]
        end = physical[node_index[target]]
        for node in nodes_plot:
            if node in (source, target):
                continue
            k = node_index[node]
            distance, _ = physical_segment_distance(physical[k], start, end)
            clearance = distance - node_radius_pt[k]
            min_edge_node_clearance_pt = min(min_edge_node_clearance_pt, clearance)
            if clearance < -1e-6:
                edge_node_intrusions += 1

    if node_overlap_count != 0:
        raise RuntimeError(f"Inter-industry network still has {node_overlap_count} rendered node overlaps")

    (RESULTS_DIR / "interindustry_dynamics_plot_layout_metrics.json").write_text(
        json.dumps({
            "plot_edges": int(len(plot_edges)),
            "plot_nodes": int(len(nodes_plot)),
            "node_overlap_count": int(node_overlap_count),
            "min_node_clearance_points": float(min_node_clearance_pt),
            "edge_node_intrusions": int(edge_node_intrusions),
            "min_edge_node_clearance_points": float(min_edge_node_clearance_pt),
            "layout_seed": 4,
        }, indent=2),
        encoding="utf-8",
    )

    pos = {
        node: (
            float(physical[node_index[node], 0] / axes_width_pt),
            float(physical[node_index[node], 1] / axes_height_pt),
        )
        for node in nodes_plot
    }

    alpha_scale = float(np.quantile(abs_beta, 0.90))
    for row in plot_edges.itertuples(index=False):
        normalized = min(row.abs_beta / alpha_scale, 1.0) if alpha_scale > 0 else 0.5
        alpha = 0.24 + 0.58 * normalized ** 0.85
        edge_color = "#477F98" if row.standardized_beta >= 0 else "#9A5F57"
        nx.draw_networkx_edges(
            plot_graph,
            pos,
            edgelist=[(row.source, row.target)],
            nodelist=[row.source, row.target],
            node_size=[node_sizes_plot[row.source], node_sizes_plot[row.target]],
            ax=ax,
            edge_color=edge_color,
            width=1.15,
            alpha=alpha,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=8.0,
            connectionstyle="arc3,rad=0.0",
        )

    ax.scatter(
        [pos[node][0] for node in nodes_plot],
        [pos[node][1] for node in nodes_plot],
        s=[node_sizes_plot[node] for node in nodes_plot],
        facecolor="#4E86A8",
        edgecolor="#2F6485",
        linewidth=0.82,
        alpha=0.96,
        zorder=4,
    )

    label_font = FontProperties(weight="bold")

    def fitting_fontsize(node_size: float, text_value: str) -> float | None:
        diameter = 2.0 * math.sqrt(node_size / math.pi)
        fontsize = 9.4
        while fontsize >= 6.0:
            bounds = TextPath((0, 0), text_value, size=fontsize, prop=label_font).get_extents()
            if bounds.width <= diameter * 0.78 and bounds.height <= diameter * 0.68:
                return fontsize
            fontsize -= 0.2
        return None

    for node in nodes_plot:
        px, py = pos[node]
        text_value = industry_code.get(node, node[:4].upper())
        fontsize = fitting_fontsize(node_sizes_plot[node], text_value)
        ax.text(
            px,
            py,
            text_value,
            ha="center",
            va="center",
            fontsize=fontsize or 6.0,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#4E86A8",
            markeredgecolor="#2F6485", markersize=6.6,
            label="Industry (size: selected-network strength)",
        ),
        Line2D([0, 1], [0, 0], color="#477F98", lw=1.35, alpha=0.75, label="Positive relation"),
        Line2D([0, 1], [0, 0], color="#9A5F57", lw=1.35, alpha=0.75, label="Negative relation"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.33, 0.995),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        title="Inter-industry ICI dynamics • arrow = lead→lag • distance/opacity encode |β|",
        title_fontsize=9.6,
        columnspacing=0.9,
        handlelength=1.45,
        handletextpad=0.35,
    )

    # Full industry names live in a dedicated right-hand panel, leaving the
    # network itself readable with compact codes only.
    code_ax = fig.add_axes([0.67, 0.075, 0.31, 0.80])
    code_ax.set_xlim(0.0, 1.0)
    code_ax.set_ylim(0.0, 1.0)
    code_ax.axis("off")
    code_ax.text(
        0.0, 0.975, "Industry codes", ha="left", va="top",
        fontsize=11.0, fontweight="bold", color="#1f2937",
    )
    code_ax.plot([0.0, 1.0], [0.935, 0.935], color="#cbd5e1", linewidth=0.8)
    code_rows = sorted(
        ((industry_code.get(node, node[:4].upper()), node_name_map[node]) for node in nodes_plot),
        key=lambda item: item[0],
    )
    y_code = 0.900
    code_step = 0.054
    for code, full_name in code_rows:
        code_ax.text(
            0.0, y_code, code,
            ha="left", va="center", fontsize=8.2, fontweight="bold", color="#2F6485",
        )
        code_ax.text(
            0.13, y_code, full_name,
            ha="left", va="center", fontsize=7.8, color="#334155",
        )
        y_code -= code_step

    ax.axis("off")
    path = PLOTS_DIR / "07_interindustry_dynamics_network.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 8) Three cyber-Figure-style ego networks for the strongest dynamic hubs.
    focal_nodes = node_summary[node_summary["selected_network_strength"] > 0].head(3)["node_key"].tolist()
    fig, axes = plt.subplots(1, max(len(focal_nodes), 1), figsize=(18, 6.5))
    if len(focal_nodes) == 1:
        axes = np.asarray([axes])
    for ax, focal in zip(np.asarray(axes).ravel(), focal_nodes):
        incident = edges[(edges["source"] == focal) | (edges["target"] == focal)].head(EGO_EDGE_COUNT * 2).copy()
        # cap each side so one direction cannot monopolize the ego plot
        outgoing = incident[incident["source"] == focal].head(EGO_EDGE_COUNT)
        incoming = incident[incident["target"] == focal].head(EGO_EDGE_COUNT)
        ego_edges = pd.concat([outgoing, incoming]).drop_duplicates(["source", "target"]).sort_values("abs_beta", ascending=False)
        eg = nx.DiGraph()
        for row in ego_edges.itertuples(index=False):
            eg.add_edge(row.source, row.target, weight=row.abs_beta, beta=row.standardized_beta)
        if focal not in eg:
            eg.add_node(focal)
        ego_pos = nx.spring_layout(eg, seed=4, k=1.05, iterations=900, weight=None)
        local_strength = {
            n: sum(abs(d["beta"]) for _, _, d in eg.in_edges(n, data=True))
            + sum(abs(d["beta"]) for _, _, d in eg.out_edges(n, data=True))
            for n in eg.nodes
        }
        local_max = max(local_strength.values()) or 1.0
        n_sizes = [
            (520 + 1350 * (local_strength[n] / local_max) ** 0.5) * (1.15 if n == focal else 1.0)
            for n in eg.nodes
        ]
        nx.draw_networkx_nodes(
            eg,
            ego_pos,
            node_size=n_sizes,
            node_color="#4E86A8",
            edgecolors="#2F6485",
            linewidths=[1.8 if n == focal else 0.8 for n in eg.nodes],
            alpha=0.96,
            ax=ax,
        )
        for u, v, d in eg.edges(data=True):
            nx.draw_networkx_edges(
                eg, ego_pos, edgelist=[(u, v)], width=1.0 + 5.0 * abs(d["beta"]),
                edge_color="#477F98" if d["beta"] >= 0 else "#9A5F57",
                alpha=0.68, arrows=True, arrowsize=11, arrowstyle="-|>", connectionstyle="arc3,rad=0.04", ax=ax,
            )
        nx.draw_networkx_labels(
            eg,
            ego_pos,
            labels={n: industry_code.get(n, n[:4].upper()) for n in eg.nodes},
            font_size=7.4,
            font_weight="bold",
            font_color="white",
            ax=ax,
        )
        ax.set_title(node_name_map[focal], fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Strongest inter-industry ICI hubs", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "08_interindustry_ego_networks.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    # 9) Mean component composition over time; illustrates what drives average ICI.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col in TARGET_COLUMNS[:3]:
        mean = qgroups[col].mean().reindex(quarter_order).to_numpy()
        ax.plot(x, mean, linewidth=1.4, label=label[col])
    ici_mean = qgroups["target_ici"].mean().reindex(quarter_order).to_numpy()
    ax.plot(x, ici_mean, linewidth=2.4, label="ICI")
    ax.axvline(quarter_order.index(VAL_START), linestyle="--", linewidth=1)
    ax.axvline(quarter_order.index(TEST_START), linestyle="--", linewidth=1)
    ax.set_xticks(tick_idx, [quarter_order[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_ylabel("Cross-industry mean score")
    ax.set_title("Average component composition and ICI dynamics")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = PLOTS_DIR / "09_component_composition_over_time.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path.relative_to(ROOT)))

    return generated


def write_split_csv(path: Path, df: pd.DataFrame) -> str:
    """Write a split CSV, but tolerate a Windows lock if the existing file matches."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return "written"
    except PermissionError:
        if not path.exists():
            raise
        existing = pd.read_csv(path, nrows=5)
        if list(existing.columns) != list(df.columns):
            raise RuntimeError(f"Locked existing file has different columns: {path}")
        # Count data rows without loading a ~50MB CSV a second time.
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            existing_rows = sum(1 for _ in f) - 1
        if existing_rows != len(df):
            raise RuntimeError(
                f"Locked existing file has wrong row count: {path} existing={existing_rows} expected={len(df)}"
            )
        print(f"    {path.name} is locked; verified existing schema/row count and kept it unchanged")
        return "locked_verified"


def assemble() -> None:
    ensure_dirs()
    require_inputs()
    for legacy in LEGACY_OUTPUTS:
        if legacy.exists():
            legacy.unlink()

    print("[1/7] Reconstructing ex-post 3 component targets + ICI")
    target, target_meta = build_targets()

    print("[2/7] Building causal/as-of target history")
    history, history_meta = build_causal_history(target_meta)

    print("[3/7] Loading all stage-02 feature channels")
    node, masks, node_feature_cols, global_feature_cols = load_features()

    print("[4/7] Joining targets, causal history, node features, and global features")
    key_cols = ["quarter", "node_key"]
    if (
        target.duplicated(key_cols).any()
        or history.duplicated(key_cols).any()
        or node.duplicated(key_cols).any()
        or masks.duplicated(key_cols).any()
    ):
        raise RuntimeError("Duplicate (quarter,node_key) rows detected")

    # Modeling starts only once the strict causal target-history information set
    # is fully available under the publication-cutoff policy.
    q0, q1 = history_meta["period"]
    node = node[(node["quarter"] >= q0) & (node["quarter"] <= q1)].copy()
    masks = masks[(masks["quarter"] >= q0) & (masks["quarter"] <= q1)].copy()
    target = target[(target["quarter"] >= q0) & (target["quarter"] <= q1)].copy()

    integrated = target.merge(history, on=key_cols, how="inner", validate="one_to_one")
    integrated = integrated.merge(node, on=key_cols, how="left", validate="one_to_one")
    global_df = pd.read_csv(GLOBAL_FEATURES, dtype={"quarter": str})
    global_df = global_df[(global_df["quarter"] >= q0) & (global_df["quarter"] <= q1)].copy()
    integrated = integrated.merge(global_df, on="quarter", how="left", validate="many_to_one")

    mask_out = target[["quarter", "node_key", "node_name"]].merge(
        masks, on=key_cols, how="left", validate="one_to_one"
    )

    expected_quarters = len(pd.period_range(MODEL_START, TEST_END, freq="Q"))
    expected_rows = expected_quarters * 56
    feature_summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    expected_features = int(feature_summary["total_initial_model_channels"])
    feature_cols = node_feature_cols + global_feature_cols
    if len(integrated) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(integrated)}")
    if len(feature_cols) != expected_features:
        raise RuntimeError(f"Expected {expected_features} feature columns, found {len(feature_cols)}")
    if integrated[TARGET_COLUMNS + TARGET_RAW_COLUMNS].isna().any().any():
        raise RuntimeError("Target columns contain missing values")
    if not np.isfinite(integrated[TARGET_COLUMNS + TARGET_RAW_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("Target columns contain non-finite values")
    history_values = integrated[HISTORY_COLUMNS + HISTORY_RAW_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(history_values).all():
        raise RuntimeError("Causal history must be finite across the full modeling period")

    # Column order is stable and explicit: identifiers -> outcome targets ->
    # causal target history -> node features -> global features.
    integrated = integrated[
        ["quarter", "node_key", "node_name"]
        + TARGET_COLUMNS
        + TARGET_RAW_COLUMNS
        + HISTORY_COLUMNS
        + HISTORY_RAW_COLUMNS
        + node_feature_cols
        + global_feature_cols
    ]
    mask_out = mask_out[["quarter", "node_key", "node_name"] + node_feature_cols]

    print("[5/7] Splitting chronologically at 70/15/15 and writing Train/Val/Test")
    train = integrated[integrated["quarter"] <= TRAIN_END].copy()
    val = integrated[(integrated["quarter"] >= VAL_START) & (integrated["quarter"] <= VAL_END)].copy()
    test = integrated[(integrated["quarter"] >= TEST_START) & (integrated["quarter"] <= TEST_END)].copy()

    # The three chronological partitions must be disjoint and exhaustive.
    split_frames = {"train": train, "val": val, "test": test}
    split_names = list(split_frames)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            if set(split_frames[left]["quarter"]) & set(split_frames[right]["quarter"]):
                raise RuntimeError(f"{left}/{right} quarter overlap")
    if sum(len(df) for df in split_frames.values()) != len(integrated):
        raise RuntimeError("Chronological split does not exhaust integrated panel")

    expected_split_rows = {
        "train": 56 * 56,  # 2006Q1-2019Q4 = 70% of 80 quarters
        "val": 12 * 56,    # 2020Q1-2022Q4 = 15%
        "test": 12 * 56,   # 2023Q1-2025Q4 = 15%
    }
    actual_split_rows = {name: len(df) for name, df in split_frames.items()}
    if actual_split_rows != expected_split_rows:
        raise RuntimeError(f"Unexpected split row counts: {actual_split_rows}")

    split_write_status = {
        "train": write_split_csv(TRAIN_CSV, train),
        "val": write_split_csv(VAL_CSV, val),
        "test": write_split_csv(TEST_CSV, test),
    }

    print("[6/7] Writing target distribution / dynamics diagnostics")
    generated_plots = save_target_diagnostics(integrated, train, val, test, target_meta)

    registry_rows = []
    for c in ["quarter", "node_key", "node_name"]:
        registry_rows.append({"column": c, "role": "IDENTIFIER", "scope": "ROW"})
    for c in TARGET_COLUMNS:
        registry_rows.append({"column": c, "role": "REFERENCE_TARGET_SCORE", "scope": "NODE"})
    for c in TARGET_RAW_COLUMNS:
        registry_rows.append({"column": c, "role": "TARGET_RAW", "scope": "NODE"})
    for c in HISTORY_COLUMNS:
        registry_rows.append({"column": c, "role": "EXPANDING_HISTORY_SCORE", "scope": "NODE"})
    for c in HISTORY_RAW_COLUMNS:
        registry_rows.append({"column": c, "role": "CAUSAL_HISTORY_RAW", "scope": "NODE"})
    for c in node_feature_cols:
        registry_rows.append({"column": c, "role": "FEATURE", "scope": "NODE"})
    for c in global_feature_cols:
        registry_rows.append({"column": c, "role": "FEATURE", "scope": "GLOBAL"})
    pd.DataFrame(registry_rows).to_csv(
        RESULTS_DIR / "integrated_column_registry.csv", index=False, encoding="utf-8-sig"
    )

    feature_values = integrated[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite_nonmissing = np.isfinite(feature_values.to_numpy(dtype=float, na_value=np.nan)) | feature_values.isna().to_numpy()
    if not finite_nonmissing.all():
        raise RuntimeError("Feature matrix contains +/-inf")

    feature_summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "train": str(TRAIN_CSV.relative_to(ROOT)),
            "val": str(VAL_CSV.relative_to(ROOT)),
            "test": str(TEST_CSV.relative_to(ROOT)),
        },
        "period": history_meta["period"],
        "quarters": expected_quarters,
        "nodes": target_meta["node_count"],
        "rows": len(integrated),
        "split_periods": {
            "train": [MODEL_START, TRAIN_END],
            "val": [VAL_START, VAL_END],
            "test": [TEST_START, TEST_END],
        },
        "split_ratio_quarters": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_rows": actual_split_rows,
        "split_write_status": split_write_status,
        "identifier_columns": 3,
        "target_columns": TARGET_COLUMNS,
        "target_count": len(TARGET_COLUMNS),
        "target_raw_columns": TARGET_RAW_COLUMNS,
        "target_raw_count": len(TARGET_RAW_COLUMNS),
        "causal_history_columns": HISTORY_COLUMNS,
        "causal_history_count": len(HISTORY_COLUMNS),
        "causal_history_raw_columns": HISTORY_RAW_COLUMNS,
        "causal_history_raw_count": len(HISTORY_RAW_COLUMNS),
        "causal_history_available_rows": int(integrated[HISTORY_COLUMNS].notna().all(axis=1).sum()),
        "causal_history_missing_rows": int(integrated[HISTORY_COLUMNS].isna().all(axis=1).sum()),
        "information_cutoff": history_meta,
        "node_feature_columns": len(node_feature_cols),
        "global_feature_columns": len(global_feature_cols),
        "total_feature_columns": len(feature_cols),
        "total_csv_columns": len(integrated.columns),
        "target_normalization": "reference scores use robust signed q05/q95 scaling with zero=50; modeling refits the same transform expanding through each forecast origin in stage 05",
        "target_denton_max_annual_benchmark_abs_error": target_meta["denton_max_annual_benchmark_abs_error"],
        "feature_pool_expected_channels": feature_summary["total_initial_model_channels"],
        "feature_missing_cells": int(feature_values.isna().sum().sum()),
        "feature_all_nan_columns": int(feature_values.isna().all().sum()),
        "target_diagnostic_plots": generated_plots,
        "status": "PASS",
    }
    (RESULTS_DIR / "integration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[7/7] PASS")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble PRISM target + all-feature quarterly CSV")
    parser.parse_args()
    assemble()


if __name__ == "__main__":
    main()
