#!/usr/bin/env python3
"""Calculate PRISM's three targets and ICI, then join all 115 source features.

Inputs are the latest (revised) official sources from Stages 01 and 02.
This stage neither retrieves data nor produces extra predictor features,
PIT histories, imputations, or fitted forecasting models.
The THREE target scores use a single pooled Train-only Min-Max reference;
predictor-feature scaling belongs to the later training pipeline.

Normalization rationale for the paper (fixed-origin experiment):
- For each of profitability, productivity and growth, calculate ONE common
  min/max from all 56 industries in the complete Train period
  (2006Q1-2019Q4), then hold those bounds fixed for Val (2020Q1-2022Q4)
  and Test (2023Q1-2025Q4). Equal raw values therefore receive equal scores
  at every date; later observations cannot rescale historical scores.
- Fitting on the whole Train set is valid for a model trained once at the end
  of 2019. Never include Val/Test observations when fitting these bounds.
  A rolling-origin historical forecast instead requires bounds refitted
  only on the data available at each earlier forecast origin.
- Scores below 0 or above 100 are mathematically expected when a held-out
  observation exceeds Train extrema. The 0-100 range is a TRAIN reference,
  not a hard limit. Do NOT clip: truncation discards genuine changes.
- Preserve the original pooled Min-Max and equal-weight ICI definition.
  Do not winsorize or apply robust/nonlinear rescaling merely because an
  extreme value exists: verified industry differences and economic shocks
  are observations, not automatically data errors. This remains sensitive
  to Train extremes; disclose that interpretation limitation. Alternative
  normalizations are optional sensitivity analyses, not a mathematical
  prerequisite for using the defined index.
- These are *split-level scaler* leakage rules, not a claim of real-time
  vintage validity: Stage 01 uses retrospectively revised official data.
- Quarterly GOS uses FIRST-DIFFERENCE PROPORTIONAL DENTON within the Train,
  Validation and Test year blocks separately. Train GOS is frozen before
  Validation is calculated; Test never changes prior GOS. The prior block's
  final GOS/indicator ratio anchors the next block to preserve smoothness.
  This avoids hidden held-out annual GOS entering TRAIN labels.

Outputs: raw_train.csv, raw_val.csv and raw_test.csv, each one row per
(quarter, node_key) with four scored targets and 115 original features.
Raw component levels and verification details are saved under Results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_01 = ROOT / "Data/01_target_data_collection/latest_sources.csv.gz"
REGISTRY_01 = ROOT / "Results/01_target_data_collection/node_registry.csv"
SUMMARY_01 = ROOT / "Results/01_target_data_collection/collection_summary.json"
NODE_02 = ROOT / "Data/02_initial_feature_pool/processed/node_features_quarterly_long.csv"
GLOBAL_02 = ROOT / "Data/02_initial_feature_pool/processed/global_features_quarterly.csv"
MASK_02 = ROOT / "Data/02_initial_feature_pool/processed/node_feature_masks_quarterly_long.csv"
SUMMARY_02 = ROOT / "Results/02_initial_feature_pool/feature_pool_summary.json"
DATA = ROOT / "Data/03_integrated_dataset"
RESULTS = ROOT / "Results/03_integrated_dataset"
MODEL_START, MODEL_END = "2006Q1", "2025Q4"
HISTORY_START = "2005Q1"  # Needed for 4-quarter real-output growth.
SPLIT_PERIODS = {
    "train": ("2006Q1", "2019Q4"),
    "val": ("2020Q1", "2022Q4"),
    "test": ("2023Q1", "2025Q4"),
}
SPLIT_FILES = {
    "train": DATA / "raw_train.csv",
    "val": DATA / "raw_val.csv",
    "test": DATA / "raw_test.csv",
}
TARGETS = ["target_profitability", "target_productivity", "target_growth", "target_ici"]
RAW_COMPONENTS = ["profitability_raw", "productivity_raw", "growth_raw"]
IDS = ["quarter", "node_key", "node_name"]


def require_files() -> None:
    required = [SOURCE_01, REGISTRY_01, SUMMARY_01, NODE_02, GLOBAL_02,
                MASK_02, SUMMARY_02]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 03 requires completed 01/02 outputs: " + "; ".join(missing))
    for summary in (SUMMARY_01, SUMMARY_02):
        obj = json.loads(summary.read_text(encoding="utf-8"))
        if obj.get("status") != "PASS" or obj.get("policy") != "LATEST_OFFICIAL_VALUES_ONLY":
            raise RuntimeError(f"Input is not the verified latest-value source: {summary}")


def load_nodes() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY_01, dtype=str).fillna("")
    if (list(registry.columns) != ["node_key", "industry_name", "naics_codes"]
            or len(registry) != 56 or registry.node_key.duplicated().any()):
        raise RuntimeError("Stage 01 must define exactly 56 unique PRISM industries")
    return registry


def load_latest_sources() -> pd.DataFrame:
    data = pd.read_csv(
        SOURCE_01, low_memory=False,
        dtype={"period": str, "node_key": str, "component": str,
               "source_industry": str, "frequency": str},
    )
    required = {"provider", "series", "frequency", "period", "node_key",
                "component", "units", "value"}
    if not required.issubset(data):
        raise RuntimeError(f"Missing source columns: {sorted(required - set(data))}")
    if data[list(required - {"value"})].isna().any().any():
        raise RuntimeError("Stage 01 has a source with missing identity or units")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if not np.isfinite(data.value.to_numpy(dtype=float)).all():
        raise RuntimeError("Stage 01 has nonfinite source values")
    identity = ["provider", "series", "frequency", "period", "node_key", "component"]
    if data.duplicated(identity).any():
        raise RuntimeError("Duplicate latest native source components")
    return data


def bea_panel(source: pd.DataFrame, series: str, frequency: str,
              periods: list[str], nodes: list[str]) -> pd.DataFrame:
    """Sum the exact BEA native components of a PRISM industry per period.

    The 01 industry mapping keeps Hospitals and Nursing as distinct components.
    Their summed chained-dollar totals are a fixed-reference-dollar
    approximation, consistent with the published PRISM 56-node definition.
    All other nodes have a single BEA component.
    """
    table = source[
        (source.provider == "BEA") & (source.series == series)
        & (source.frequency == frequency) & source.period.isin(periods)
    ]
    expected = pd.MultiIndex.from_product([periods, nodes], names=["period", "node_key"])
    groups = table.groupby(["period", "node_key"], sort=False)
    component_count = groups.component.nunique().reindex(expected)
    expected_count = pd.Series(
        [2 if node == "hospitals_nursing" else 1 for _, node in expected],
        index=expected,
    )
    if not component_count.eq(expected_count).all():
        missing = expected[~component_count.eq(expected_count)]
        raise RuntimeError(f"Missing BEA {series}/{frequency} native components: {list(missing[:8])}")
    # Annual and quarterly units are deliberately not mixed; separately
    # validate the amount scale and the chain-dollar reference year.
    units = table.units.unique()
    if len(units) != 1:
        raise RuntimeError(f"Incompatible {series}/{frequency} units: {units}")
    actual = groups.value.sum().reindex(expected)
    if actual.isna().any():
        raise RuntimeError(f"Incomplete BEA {series}/{frequency} panel")
    return actual.unstack("node_key").reindex(index=periods, columns=nodes).astype(float)


def employment_panel(source: pd.DataFrame, periods: list[str], nodes: list[str]) -> pd.DataFrame:
    """Three monthly official employment counts -> one quarterly mean.

    QCEW is stored in people and CES rail employment is in thousands of people.
    The resulting denominator is thousands of employed people everywhere.
    """
    employment = source[source.series == "employment"].copy()
    wrong_units = ~employment.units.isin({"persons", "thousands_of_people"})
    if wrong_units.any():
        raise RuntimeError("Unrecognized employment unit in Stage 01")
    employment["thousands"] = np.where(
        employment.units == "persons", employment.value / 1000.0, employment.value)
    employment["reference_quarter"] = pd.PeriodIndex(
        employment.period, freq="M").asfreq("Q").astype(str)
    months = [str(month) for month in pd.period_range(
        pd.Period(periods[0], freq="Q").asfreq("M", "start"),
        pd.Period(periods[-1], freq="Q").asfreq("M", "end"), freq="M")]
    required = pd.MultiIndex.from_product([months, nodes], names=["period", "node_key"])
    counts = employment.groupby(["period", "node_key"]).component.nunique().reindex(required)
    registry = pd.read_csv(REGISTRY_01, dtype=str).fillna("")
    required_parts = {r.node_key: (1 if r.node_key == "rail_transport"
                                   else len(r.naics_codes.split(",")))
                      for r in registry.itertuples(index=False)}
    expected_counts = pd.Series([required_parts[node] for _, node in required], index=required)
    if not counts.eq(expected_counts).all():
        missing = required[~counts.eq(expected_counts)]
        raise RuntimeError(f"Incomplete industry employment native components: {list(missing[:8])}")
    monthly = (employment.groupby(["period", "node_key"]).thousands.sum()
               .reindex(required))
    if monthly.isna().any() or (monthly <= 0).any():
        raise RuntimeError("Employment denominator contains missing or nonpositive observations")
    monthly = monthly.rename("employment_thousands").reset_index()
    monthly["quarter"] = pd.PeriodIndex(monthly.period, freq="M").asfreq("Q").astype(str)
    avg = monthly.groupby(["quarter", "node_key"]).employment_thousands.agg(["mean", "count"])
    if (avg["count"] != 3).any():
        raise RuntimeError("Industry quarterly employment must have all three months")
    out = avg["mean"].unstack("node_key").reindex(index=periods, columns=nodes)
    if out.isna().any().any():
        raise RuntimeError("Incomplete 56-industry quarterly employment panel")
    return out.astype(float)


FUNDS_AGGREGATE_CES_ID = "CEU5552300001"
FUNDS_AGGREGATE_SOURCE = "employment_523_525_aggregate"


def funds_aggregate_employment(source: pd.DataFrame, quarters: list[str]) -> pd.Series:
    """Latest CES NSA 523+525 employment, with complete three-month quarters.

    Securities (523) and funds/trusts (525) are grouped in this CES series.
    This denominator is paired with the *same combined* BEA real-VA numerator
    for the funds/trusts productivity target only.
    """
    monthly = source[
        (source.provider == "BLS_CES")
        & (source.node_key == "funds_trusts")
        & (source.series == FUNDS_AGGREGATE_SOURCE)
    ].copy()
    if (monthly.empty
            or not monthly.frequency.eq("monthly").all()
            or not monthly.source_industry.eq(FUNDS_AGGREGATE_CES_ID).all()
            or not monthly.units.eq("thousands_of_people").all()
            or monthly.period.duplicated().any()):
        raise RuntimeError("Missing or incompatible CES 523+525 employment source")
    months = pd.period_range(
        pd.Period(quarters[0], freq="Q").asfreq("M", "start"),
        pd.Period(quarters[-1], freq="Q").asfreq("M", "end"),
        freq="M",
    ).astype(str)
    values = monthly.set_index("period").value.reindex(months)
    if values.isna().any() or (values <= 0).any():
        raise RuntimeError("CES 523+525 monthly employment is incomplete")
    ref_quarters = pd.PeriodIndex(months, freq="M").asfreq("Q").astype(str)
    grouped = values.groupby(ref_quarters)
    quarterly = grouped.mean().where(grouped.count() == 3).reindex(quarters)
    if quarterly.isna().any():
        raise RuntimeError("CES 523+525 quarter lacks all three published months")
    return quarterly.rename("ces_523_525_employment_thousands")


def proportional_denton(indicator: np.ndarray, annual_benchmark: np.ndarray,
                        previous_ratio: float | None = None) -> np.ndarray:
    """First-difference proportional Denton; quarterly BEA figures are SAAR.

    Reconstruct quarterly GOS from quarterly current VA while requiring that
    each four-quarter mean equals the official annual GOS benchmark.
    For a later block, penalize the difference between its first quarterly
    indicator ratio and the preceding *frozen* block's last ratio.
    """
    x = np.asarray(indicator, dtype=float)
    annual = np.asarray(annual_benchmark, dtype=float)
    t, ny = len(x), len(annual)
    if t != 4 * ny or not np.isfinite(x).all() or not np.isfinite(annual).all():
        raise ValueError("Denton indicator/annual benchmark lengths or values invalid")
    if (np.abs(x) < 1e-12).any():
        raise ValueError("Denton indicator contains an exact zero")
    difference = np.diff(np.eye(t), axis=0)
    penalty = difference.T @ difference + np.eye(t) * 1e-10
    rhs_ratio = np.zeros(t, dtype=float)
    if previous_ratio is not None:
        if not np.isfinite(previous_ratio):
            raise ValueError("Nonfinite preceding-quarter Denton ratio")
        penalty[0, 0] += 1.0
        rhs_ratio[0] = previous_ratio
    constraints = np.zeros((ny, t), dtype=float)
    for year in range(ny):
        constraints[year, 4 * year:4 * (year + 1)] = x[4 * year:4 * (year + 1)]
    system = np.block([[penalty, constraints.T], [constraints, np.zeros((ny, ny))]])
    rhs = np.concatenate([rhs_ratio, 4.0 * annual])
    quarterly = x * np.linalg.solve(system, rhs)[:t]
    if not np.isfinite(quarterly).all():
        raise RuntimeError("Nonfinite quarterly Denton GOS estimates")
    if not np.allclose(quarterly.reshape(ny, 4).mean(axis=1), annual,
                       atol=1e-5, rtol=1e-10):
        raise RuntimeError("Denton annual GOS benchmarks not preserved")
    return quarterly


def denton_with_frozen_split_boundaries(
    indicator: np.ndarray, annual_benchmark: np.ndarray,
) -> np.ndarray:
    """Benchmarked GOS with no held-out annual information entering Train.

    Train includes 2005 (needed for 2006 onward calculations) through 2019.
    Val has 2020-2022; Test has 2023-2025. At each boundary the preceding
    GOS is FROZEN, while its final GOS/VA ratio anchors the next block's
    first-difference Denton optimization. This retains the annual benchmarks
    and avoids artificial independent-block resets of the ratio penalty.

    The ex-post labels use official annual benchmarks of their OWN periods,
    not information available at every historical forecast origin.
    """
    x = np.asarray(indicator, dtype=float)
    annual = np.asarray(annual_benchmark, dtype=float)
    first_year = int(HISTORY_START[:4])
    last_year = int(MODEL_END[:4])
    if (len(x) != 4 * (last_year - first_year + 1)
            or len(annual) != last_year - first_year + 1):
        raise ValueError("Denton input does not match the required complete annual history")
    result = np.empty_like(x)
    begin_year = first_year
    previous_ratio = None
    for _, (_, end_quarter) in SPLIT_PERIODS.items():
        end_year = int(end_quarter[:4])
        if end_year < begin_year:
            raise RuntimeError("Denton split boundaries are not chronological")
        start = 4 * (begin_year - first_year)
        stop = 4 * (end_year - first_year + 1)
        chunk = proportional_denton(
            x[start:stop], annual[start // 4:stop // 4],
            previous_ratio=previous_ratio,
        )
        result[start:stop] = chunk
        previous_ratio = float(chunk[-1] / x[stop - 1])
        begin_year = end_year + 1
    if begin_year != last_year + 1:
        raise RuntimeError("Denton split boundaries do not cover all target years")
    if not np.allclose(result.reshape(-1, 4).mean(axis=1), annual,
                       atol=1e-5, rtol=1e-10):
        raise RuntimeError("Split-frozen Denton failed annual benchmarks")
    return result


def score_with_train_reference(values: np.ndarray, quarters: list[str],
                               component: str) -> tuple[np.ndarray, list[dict]]:
    """Freeze one pooled 56-industry Min-Max reference over *all* Train quarters.

    LEAKAGE CONTRACT (single fixed-origin Train/Val/Test experiment):
    - The model is trained once with the complete 2006Q1–2019Q4 Train set.
      All 56 industries and all 56 Train quarters are available when its
      scaler is fitted; using later TRAIN observations to score earlier TRAIN
      targets is not a Train/Validation/Test split leak in this design.
    - Fit min/max ONLY on Train and use those very same two constants for
      every Train, Validation and Test row. NEVER refit on held-out target
      values or expand bounds with future quarters.
    - Do NOT clip scores to [0, 100]: values beyond Train extrema carry real
      out-of-range information. ICI is still the mean of the three scores.
    - This contract does NOT apply to historical rolling-origin backtests:
      at an earlier forecast origin (e.g. 2010), refit on observations
      available by that origin, not on all of Train through 2019Q4.
    - Separate issue: Stage 01 uses current revised official series, and
      annual-GOS Denton disaggregation uses whole-period annual benchmarks.
      Freezing the target scale does NOT turn those retrospective values
      into point-in-time data. Do not claim a real-time vintage backtest.
    """
    matrix = np.asarray(values, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != len(quarters)
            or not np.isfinite(matrix).all()):
        raise ValueError(f"Nonfinite or misaligned raw target: {component}")
    train_start, train_end = SPLIT_PERIODS["train"]
    train_quarters = pd.period_range(train_start, train_end, freq="Q").astype(str).tolist()
    if (quarters[:len(train_quarters)] != train_quarters
            or quarters != sorted(set(quarters))):
        raise ValueError(f"{component} needs an ordered complete Train prefix before Val/Test")
    # This slice is fixed ONCE; do not move it with the scored quarter.
    train_values = matrix[:len(train_quarters)]
    lo, hi = float(train_values.min()), float(train_values.max())
    if not lo < hi:
        raise RuntimeError(f"Degenerate TRAIN score bounds for {component}")
    scores = 100.0 * (matrix - lo) / (hi - lo)
    history = []
    for quarter in quarters:
        history.append({"quarter": quarter, "component": component,
                        "reference_start": train_start, "reference_end": train_end,
                        "min": lo, "max": hi,
                        "reference_rows": int(train_values.size),
                        "fixed_train_reference": True})
    return scores, history


def build_targets(source: pd.DataFrame, registry: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Only four scored targets enter the final modeling dataset."""
    nodes = registry.node_key.tolist()
    all_quarters = pd.period_range(HISTORY_START, MODEL_END, freq="Q").astype(str).tolist()
    target_quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str).tolist()
    years = [str(year) for year in range(2005, 2026)]
    current_va = bea_panel(source, "current_va", "quarterly", all_quarters, nodes)
    current_go = bea_panel(source, "current_go", "quarterly", all_quarters, nodes)
    real_va = bea_panel(source, "real_va", "quarterly", all_quarters, nodes)
    real_go = bea_panel(source, "real_go", "quarterly", all_quarters, nodes)
    gos_annual = bea_panel(source, "gos", "annual", years, nodes)
    employment = employment_panel(source, all_quarters, nodes)
    combined_ces_employment = funds_aggregate_employment(source, all_quarters)

    if (current_go.to_numpy() <= 0).any() or (real_go.to_numpy() <= 0).any():
        raise RuntimeError("Nonpositive gross-output denominator")
    gos = np.column_stack([
        denton_with_frozen_split_boundaries(
            current_va[node].to_numpy(), gos_annual[node].to_numpy()
        )
        for node in nodes
    ])
    gos_audit = []
    for j, node in enumerate(nodes):
        for y, year in enumerate(years):
            reconstructed = float(gos[y * 4:(y + 1) * 4, j].mean())
            observed = float(gos_annual.loc[year, node])
            gos_audit.append({"node_key": node, "year": year,
                              "annual_gos": observed, "quarterly_mean_gos": reconstructed,
                              "abs_error": abs(reconstructed - observed)})

    # Restore the original 523+525 matched-scope productivity treatment.
    # Standalone QCEW 525 is too small around the BLS 2013 reclassification.
    # Pair combined 523+525 CES employment with combined 523+525 BEA real VA.
    # Every OTHER industry continues using its unchanged real VA / employment.
    real_va_matched = real_va.to_numpy(dtype=float).copy()
    employment_matched = employment.to_numpy(dtype=float).copy()
    funds_idx = nodes.index("funds_trusts")
    securities_idx = nodes.index("securities")
    real_va_matched[:, funds_idx] += real_va_matched[:, securities_idx]
    employment_matched[:, funds_idx] = combined_ces_employment.to_numpy(dtype=float)
    funds_correction = pd.DataFrame({
        "quarter": target_quarters,
        "qcew_525_employment_thousands": employment.iloc[4:, funds_idx].to_numpy(),
        "ces_523_525_employment_thousands": combined_ces_employment.iloc[4:].to_numpy(),
        "bea_funds_real_va_million_2017_dollars": real_va.iloc[4:, funds_idx].to_numpy(),
        "bea_securities_real_va_million_2017_dollars": real_va.iloc[4:, securities_idx].to_numpy(),
        "bea_combined_real_va_million_2017_dollars": real_va_matched[4:, funds_idx],
        "old_unmatched_productivity": (
            real_va.iloc[4:, funds_idx].to_numpy()
            / employment.iloc[4:, funds_idx].to_numpy()),
        "restored_matched_productivity": (
            real_va_matched[4:, funds_idx] / employment_matched[4:, funds_idx]),
    })

    raw_values = {
        "profitability": (gos / current_go.to_numpy(dtype=float))[4:],
        "productivity": (real_va_matched / employment_matched)[4:],
        "growth": (real_go.to_numpy(dtype=float)[4:]
                   / real_go.to_numpy(dtype=float)[:-4] - 1.0),
    }
    scored, bounds = {}, []
    for label, values in raw_values.items():
        scored[label], history = score_with_train_reference(values, target_quarters, label)
        bounds.extend(history)
    ici = (scored["profitability"] + scored["productivity"] + scored["growth"]) / 3.0

    index = pd.MultiIndex.from_product(
        [target_quarters, nodes], names=["quarter", "node_key"])
    target = pd.DataFrame({
        "target_profitability": scored["profitability"].reshape(-1),
        "target_productivity": scored["productivity"].reshape(-1),
        "target_growth": scored["growth"].reshape(-1),
        "target_ici": ici.reshape(-1),
    }, index=index).reset_index()
    target.insert(2, "node_name", target.node_key.map(
        dict(zip(registry.node_key, registry.industry_name))))
    raw_frame = pd.DataFrame({
        "profitability_raw": raw_values["profitability"].reshape(-1),
        "productivity_raw": raw_values["productivity"].reshape(-1),
        "growth_raw": raw_values["growth"].reshape(-1),
    }, index=index).reset_index()
    return target, raw_frame, pd.DataFrame(bounds), pd.DataFrame(gos_audit), funds_correction


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    node = pd.read_csv(NODE_02, dtype={"quarter": str, "node_key": str}, low_memory=False)
    global_features = pd.read_csv(GLOBAL_02, dtype={"quarter": str}, low_memory=False)
    masks = pd.read_csv(MASK_02, dtype={"quarter": str, "node_key": str}, low_memory=False)
    node_columns = [col for col in node if col not in {"quarter", "node_key"}]
    global_columns = [col for col in global_features if col != "quarter"]
    summary = json.loads(SUMMARY_02.read_text(encoding="utf-8"))
    expected_node, expected_global = int(summary["node_features"]), int(summary["global_features"])
    if (len(node_columns) != expected_node or len(global_columns) != expected_global
            or len(node_columns) + len(global_columns) != summary["total_original_features"]
            or len(node_columns) + len(global_columns) != 115):
        raise RuntimeError("Stage 02 no longer has the expected 115 original feature channels")
    if set(node_columns) & set(global_columns):
        raise RuntimeError("Stage 02 node/global feature names overlap")
    if masks.columns.tolist() != node.columns.tolist():
        raise RuntimeError("Stage 02 industry masks and features have different columns")
    if not node[["quarter", "node_key"]].equals(masks[["quarter", "node_key"]]):
        raise RuntimeError("Stage 02 node masks are not aligned with their features")
    if not np.array_equal(node[node_columns].notna().to_numpy(),
                          masks[node_columns].to_numpy(dtype=bool)):
        raise RuntimeError("Stage 02 node masks disagree with original data")
    for table, keys in ((node, ["quarter", "node_key"]),
                        (global_features, ["quarter"])):
        if table.duplicated(keys).any():
            raise RuntimeError(f"Duplicate feature identifiers: {keys}")
    for frame, columns in ((node, node_columns), (global_features, global_columns)):
        numeric = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise RuntimeError("Stage 02 contains an infinite feature value")
    return node, global_features, node_columns, global_columns


def save_target_mean_timeseries(integrated: pd.DataFrame) -> dict:
    """Plot the equal-weight 56-industry quarterly mean of all four targets.

    Recalculate directly from the assembled scored targets, not from raw axes
    or a rolling average. This graph is a diagnostic and changes no model data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if integrated.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError("Duplicate industry-quarter targets in mean time series")
    means = integrated.groupby("quarter", sort=True)[TARGETS].agg(["mean", "count"])
    counts = means.xs("count", axis=1, level=1)
    if not counts.eq(56).all().all():
        raise RuntimeError("Target mean plot requires exactly 56 industries per quarter")
    means = means.xs("mean", axis=1, level=1).reindex(
        pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str)
    )
    if means.shape != (80, 4) or not np.isfinite(means.to_numpy(float)).all():
        raise RuntimeError("Quarterly mean targets are incomplete or nonfinite")
    if not np.allclose(means["target_ici"], means[TARGETS[:3]].mean(axis=1),
                       atol=1e-12, rtol=1e-12):
        raise RuntimeError("Quarterly mean ICI differs from the mean of three axes")

    data_file = RESULTS / "target_quarterly_mean_timeseries.csv"
    figure_file = RESULTS / "target_quarterly_mean_timeseries.png"
    means.rename_axis("quarter").reset_index().to_csv(
        data_file, index=False, encoding="utf-8-sig")

    series = (
        ("target_profitability", "Profitability", "#2878A9"),
        ("target_productivity", "Productivity", "#E18C36"),
        ("target_growth", "Growth", "#7E60A7"),
    )
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=160)
    x = np.arange(len(means))
    try:
        for column, label, color in series:
            ax.plot(x, means[column].to_numpy(), label=label, color=color,
                    linewidth=1.7, alpha=0.9)
        ax.plot(x, means["target_ici"].to_numpy(), label="ICI", color="#202B38",
                linewidth=2.7, linestyle="-")
        ax.set_title("PRISM | Quarterly Mean Target Scores", fontsize=17, pad=15,
                     fontweight="bold")
        ax.text(0.0, 1.015, "56-industry mean; fixed 2006Q1–2019Q4 Train reference",
                transform=ax.transAxes, fontsize=10, color="#566474")
        ax.set_ylabel("Mean score (Train-reference units)")
        ax.set_xlabel("Quarter")
        # A fixed Train reference permits future scores outside [0, 100].
        # Avoid silently hiding these legitimate observations in the plot.
        ymin = float(means.to_numpy(float).min())
        ymax = float(means.to_numpy(float).max())
        margin = max(2., (ymax - ymin) * 0.08)
        ax.set_ylim(min(0., ymin - margin), max(100., ymax + margin))
        ticks = list(range(0, len(means), 8))
        ax.set_xticks(ticks, [means.index[i][:4] for i in ticks])
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  frameon=False, fontsize=10)
        fig.subplots_adjust(bottom=0.2, top=0.83, left=0.075, right=0.975)
        fig.savefig(figure_file, dpi=180, facecolor="white")
    finally:
        plt.close(fig)
    return {
        "plot": figure_file.relative_to(ROOT).as_posix(),
        "quarterly_means_csv": data_file.relative_to(ROOT).as_posix(),
        "mean_method": "arithmetic mean over all 56 industries, separately for each quarter and scored target",
        "quarters": len(means), "industries_per_quarter": 56,
    }


def split_integrated_panel(integrated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Restore the user's exact raw_{train,val,test}.csv chronological split.

    Keep every source feature, target and missing-value mask as they were
    computed. Stage 03 only partitions rows, without rescaling or imputing.
    """
    if integrated.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError("Duplicate industry-quarter observations before split")
    full_quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str).tolist()
    observed_quarters = sorted(integrated["quarter"].unique())
    if observed_quarters != full_quarters:
        raise RuntimeError("Integrated panel does not cover the exact 2006Q1–2025Q4 period")

    splits: dict[str, pd.DataFrame] = {}
    claimed: set[str] = set()
    for name, (start, end) in SPLIT_PERIODS.items():
        quarters = pd.period_range(start, end, freq="Q").astype(str).tolist()
        if claimed.intersection(quarters):
            raise RuntimeError("Train/Validation/Test quarter ranges overlap")
        claimed.update(quarters)
        part = integrated.loc[integrated["quarter"].isin(quarters)].copy()
        if (len(part) != len(quarters) * 56
                or part["quarter"].nunique() != len(quarters)
                or not part.groupby("quarter")["node_key"].nunique().eq(56).all()
                or part["quarter"].tolist() != sorted(part["quarter"].tolist())
                or part.columns.tolist() != integrated.columns.tolist()):
            raise RuntimeError(f"Invalid {name} split coverage, ordering, or column schema")
        splits[name] = part.reset_index(drop=True)

    if claimed != set(full_quarters):
        raise RuntimeError("Chronological split does not exhaust the modeling quarters")
    restored = pd.concat(splits.values(), ignore_index=True)
    if not restored.equals(integrated.reset_index(drop=True)):
        raise RuntimeError("Splitting changed, dropped, or reordered input values")
    return splits


def assemble() -> dict:
    require_files()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    registry = load_nodes()
    source = load_latest_sources()

    print("[1/4] Calculating profitability, productivity, growth and ICI", flush=True)
    targets, raw, score_bounds, denton_check, funds_correction = build_targets(source, registry)
    print("[2/4] Joining the 115 original Stage 02 features", flush=True)
    node, glob, node_features, global_features = load_features()
    if targets.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError("Duplicate target rows")
    joined = targets.merge(node, on=["quarter", "node_key"], how="left",
                           validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("A target row is missing Stage 02 industry features")
    joined = joined.drop(columns="_merge")
    joined = joined.merge(glob, on="quarter", how="left", validate="many_to_one",
                          indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("A target row is missing Stage 02 global features")
    joined = joined.drop(columns="_merge")
    columns = IDS + TARGETS + node_features + global_features
    joined = joined[columns].sort_values(["quarter", "node_key"], kind="stable").reset_index(drop=True)
    expected_quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str)
    if (len(joined) != 80 * 56 or joined.node_key.nunique() != 56
            or joined.quarter.nunique() != 80 or len(joined.columns) != 122
            or len(node_features) + len(global_features) != 115
            or not joined.groupby("quarter").node_key.nunique().eq(56).all()
            or set(joined.quarter) != set(expected_quarters)):
        raise RuntimeError("Incomplete 80-quarter x 56-industry integration")
    components = joined[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(components).all():
        raise RuntimeError("The three axes or ICI contain missing or infinite values")
    if not np.allclose(components[:, 3], components[:, :3].mean(axis=1), atol=1e-10):
        raise RuntimeError("ICI is not the arithmetic mean of the three target scores")
    features = joined[node_features + global_features]
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise RuntimeError("Input features contain infinite values")

    print("[3/4] Writing raw_train.csv / raw_val.csv / raw_test.csv and target audits", flush=True)
    splits = split_integrated_panel(joined)
    for name, part in splits.items():
        part.to_csv(SPLIT_FILES[name], index=False, encoding="utf-8-sig")
    raw.to_csv(RESULTS / "target_raw_components.csv", index=False, encoding="utf-8-sig")
    score_bounds.to_csv(RESULTS / "target_score_bounds.csv", index=False, encoding="utf-8-sig")
    denton_check.to_csv(RESULTS / "gos_denton_benchmark_audit.csv", index=False,
                        encoding="utf-8-sig")
    funds_correction.to_csv(
        RESULTS / "funds_trusts_productivity_correction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"column": name, "role": ("IDENTIFIER" if name in IDS else
                                  "TARGET" if name in TARGETS else "FEATURE"),
         "scope": ("NODE" if name in node_features else
                   "GLOBAL" if name in global_features else "ROW")}
        for name in joined.columns
    ]).to_csv(RESULTS / "integrated_column_registry.csv", index=False, encoding="utf-8-sig")
    print("[4/4] Plotting the four targets' quarterly 56-industry means", flush=True)
    target_mean_timeseries = save_target_mean_timeseries(joined)
    report = {
        "status": "PASS", "policy": "LATEST_OFFICIAL_VALUES_ONLY",
        "outputs": {key: path.relative_to(ROOT).as_posix()
                    for key, path in SPLIT_FILES.items()},
        "period": [MODEL_START, MODEL_END], "quarters": 80,
        "industries": 56, "rows": len(joined), "columns": len(joined.columns),
        "split_periods": {key: list(periods)
                          for key, periods in SPLIT_PERIODS.items()},
        "split_rows": {key: len(part) for key, part in splits.items()},
        "split_ratio_quarters": {
            key: part["quarter"].nunique() / len(expected_quarters)
            for key, part in splits.items()
        },
        "identifier_columns": IDS, "target_columns": TARGETS,
        "target_count": len(TARGETS),
        "node_feature_count": len(node_features),
        "global_feature_count": len(global_features),
        "total_feature_count": len(node_features) + len(global_features),
        "auto_generated_feature_count": 0,
        "missing_targets": int(joined[TARGETS].isna().sum().sum()),
        "missing_original_feature_cells": int(features.isna().sum().sum()),
        "gos_quarterly_method": (
            "first-difference proportional Denton, current-VA SAAR indicator; "
            "Train/Val/Test annual blocks frozen sequentially, with boundary-ratio anchor"
        ),
        "gos_benchmark_blocks": {
            "train": ["2005", "2019"], "val": ["2020", "2022"],
            "test": ["2023", "2025"],
        },
        "max_gos_annual_benchmark_error": float(denton_check.abs_error.max()),
        "raw_axis_definitions": {
            "profitability": "quarterly imputed GOS / quarterly nominal gross output",
            "productivity": "quarterly real value added (million 2017 dollars, SAAR) / quarterly mean employment (thousand people); funds_trusts uses BEA 523+525 VA / CES 523+525 employment",
            "growth": "quarterly real gross output / same quarter previous year - 1",
        },
        "target_scores": (
            "fixed Train-only pooled 56-industry Min-Max, "
            "2006Q1–2019Q4; held-out Val/Test use unchanged bounds, without clipping"
        ),
        "target_normalization_fit_period": list(SPLIT_PERIODS["train"]),
        "target_normalization_clipping": False,
        "funds_trusts_productivity_correction": {
            "source": "BLS CES CEU5552300001 (NAICS 523+525 combined, NSA)",
            "numerator": "BEA securities real VA + BEA funds/trusts real VA",
            "denominator": "CES 523+525 three-month-average employment in thousands",
            "corrected_quarters": len(funds_correction),
            "audit_file": "Results/03_integrated_dataset/funds_trusts_productivity_correction.csv",
        },
        "ici": "(target_profitability + target_productivity + target_growth) / 3",
        "target_mean_timeseries": target_mean_timeseries,
        "no_imputation_or_feature_selection_in_stage_03": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "integration_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # The requested final data products are the three named split files.
    # Drop the obsolete temporary one-file output after successful generation.
    old_combined = DATA / "prism_integrated_quarterly.csv"
    if old_combined.is_file():
        old_combined.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    assemble()
