#!/usr/bin/env python3
"""Calculate PRISM targets and a compact strictly lagged feature bank.

Inputs are the latest (revised) official sources from Stages 01 and 02.
Target-derived features at row t use ONLY information through t-1,
including validated 2005 inputs where the native components exist.
Profitability history uses a past-year-annual-GOS proxy rather than the
retrospectively Denton-benchmarked Target, whose earlier quarters can depend
on later annual GOS within the same held-out split.
No same-quarter/future Targets, feature selection, imputation, external source
or point-in-time vintage is introduced by feature engineering.
The THREE target scores use a single pooled Train-only Min-Max reference;
predictor-feature scaling belongs to the later training pipeline.

Normalization rationale for the paper (fixed-origin experiment):
- For each of profitability, productivity and growth, calculate ONE common
  min/max from all 59 industries in the complete Train period
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
(quarter, node_key) with four scored targets, the strictly historical
target-feature bank first and all 115 original Stage 02 features unchanged.
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
NODE_COUNT = 59
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
HISTORY_COLUMNS = [
    "history_profitability", "history_productivity", "history_growth",
]
# The three direct lags are listed first ONLY for readability and stable
# indexing in 03-05. They have no mandatory-inclusion status in 06/07
# feature selection; any of the 157 input features may be excluded.
TARGET_AXES = ("profitability", "productivity", "growth")
TARGET_LAGS = (2, 4, 8, 12)
TARGET_WINDOWS = (4, 8)
TARGET_CHANGES = (1, 4)


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
        expected_nodes = obj.get("industry_nodes") if summary == SUMMARY_01 else obj.get("nodes")
        if expected_nodes != NODE_COUNT:
            raise RuntimeError(f"Upstream 59-node contract differs: {summary}")


def load_nodes() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY_01, dtype=str).fillna("")
    if (list(registry.columns) != ["node_key", "industry_name", "naics_codes"]
            or len(registry) != NODE_COUNT or registry.node_key.duplicated().any()
            or not registry[["node_key", "industry_name", "naics_codes"]].ne("").all().all()):
        raise RuntimeError("Stage 01 must define exactly 59 unique PRISM industries")
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
    approximation, consistent with the published PRISM 59-node definition.
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
        raise RuntimeError("Incomplete 59-industry quarterly employment panel")
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
    """Freeze one pooled 59-industry Min-Max reference over *all* Train quarters.

    LEAKAGE CONTRACT (single fixed-origin Train/Val/Test experiment):
    - The model is trained once with the complete 2006Q1–2019Q4 Train set.
      All 59 industries and all 56 Train quarters are available when its
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


def causal_history_targets(source: pd.DataFrame, targets: pd.DataFrame,
                           bounds: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Provisional unfinished-year scores, never replacements for Target labels.

    For quarter q in year y, estimate quarterly GOS using q's current VA and
    the annual GOS / quarterly-VA ratio of year y-1. Neither annual GOS of
    year y nor future-year indicators can influence this estimate.
    asof_profitability_history supersedes these provisional values whenever
    year y is already complete at the forecast origin. All 14 profitability
    features must use that same origin-specific snapshot.
    Keep the original Train-fitted profitability score scale.
    Productivity and growth already use native same/prior-quarter components.
    """
    quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str).tolist()
    periods = pd.period_range(HISTORY_START, MODEL_END, freq="Q").astype(str).tolist()
    va = bea_panel(source, "current_va", "quarterly", periods, nodes)
    gross = bea_panel(source, "current_go", "quarterly", periods, nodes)
    gos = bea_panel(source, "gos", "annual",
                    [str(year) for year in range(2005, 2025)], nodes)
    ratio = np.empty((len(quarters), len(nodes)), dtype=float)
    for year in range(2006, 2026):
        prior_year = str(year - 1)
        prior_quarters = [f"{prior_year}Q{q}" for q in range(1, 5)]
        current_quarters = [f"{year}Q{q}" for q in range(1, 5)]
        prior_va = va.loc[prior_quarters].mean(axis=0).to_numpy()
        if not np.isfinite(prior_va).all() or np.any(prior_va == 0):
            raise RuntimeError("Invalid previous-year value-added denominator")
        prev_annual_ratio = gos.loc[prior_year].to_numpy() / prior_va
        first = (year - 2006) * 4
        ratio[first:first + 4] = (
            va.loc[current_quarters].to_numpy() *
            prev_annual_ratio[None] / gross.loc[current_quarters].to_numpy())

    reference = bounds.loc[bounds.component == "profitability", ["min", "max"]]
    if (reference.empty or reference["min"].nunique() != 1
            or reference["max"].nunique() != 1):
        raise RuntimeError("Missing fixed Train profitability score bounds")
    low, high = reference.iloc[0].to_numpy(float)
    if not np.isfinite(ratio).all() or not low < high:
        raise RuntimeError("Nonfinite causal profitability history")
    index = pd.MultiIndex.from_product(
        [quarters, nodes], names=["quarter", "node_key"])
    reconstructed = targets.set_index(["quarter", "node_key"]).reindex(index).copy()
    reconstructed["target_profitability"] = (
        100. * (ratio.reshape(-1) - low) / (high - low))
    reconstructed["target_ici"] = reconstructed[
        ["target_profitability", "target_productivity", "target_growth"]].mean(axis=1)
    return reconstructed.reset_index()[IDS + TARGETS]


def pretrain_history_inputs(source: pd.DataFrame, bounds: pd.DataFrame,
                            nodes: list[str]) -> pd.DataFrame:
    """Recover usable 2005 history without inventing 2004 quarterly growth.

    The target panel intentionally starts in 2006, but Stage01 already holds
    2005 quarterly VA/GO/employment and completed 2005 *annual* GOS. These
    observations are used ONLY for predictions starting in 2006, so a
    one-year 2005 Denton allocation uses no future reference period.
    Productivity is directly reconstructible in 2005.
    Exact 2005 year-over-year growth requires 2004 quarterly real GO, which
    the 01 collector does not provide; those entries correctly remain NaN.
    """
    periods = [f"2005Q{q}" for q in range(1, 5)]
    annual_gos = bea_panel(source, "gos", "annual", ["2005"], nodes).iloc[0]
    va = bea_panel(source, "current_va", "quarterly", periods, nodes).to_numpy()
    go = bea_panel(source, "current_go", "quarterly", periods, nodes).to_numpy()
    real_va = bea_panel(source, "real_va", "quarterly", periods, nodes).to_numpy()
    employed = employment_panel(source, periods, nodes).to_numpy()
    funds, securities = nodes.index("funds_trusts"), nodes.index("securities")
    real_va[:, funds] += real_va[:, securities]
    employed[:, funds] = funds_aggregate_employment(source, periods).to_numpy()
    reconstructed_gos = np.column_stack([
        proportional_denton(va[:, j], np.asarray([annual_gos.iloc[j]], float))
        for j in range(NODE_COUNT)])
    raw_profit = reconstructed_gos / go
    raw_productivity = real_va / employed
    score = []
    for axis, values in (("profitability", raw_profit),
                         ("productivity", raw_productivity)):
        reference = bounds.loc[bounds.component.eq(axis), ["min", "max"]]
        if (reference.empty or reference["min"].nunique() != 1
                or reference["max"].nunique() != 1):
            raise RuntimeError(f"Train-only score reference missing for 2005 {axis}")
        low, high = reference.iloc[0].to_numpy(float)
        if not np.isfinite(values).all() or not low < high:
            raise RuntimeError(f"Invalid pre-2006 {axis} history")
        score.append(100. * (values - low) / (high - low))
    index = pd.MultiIndex.from_product(
        [periods, nodes], names=["quarter", "node_key"])
    return pd.DataFrame({
        "quarter": index.get_level_values("quarter"),
        "node_key": index.get_level_values("node_key"),
        "target_profitability": score[0].reshape(-1),
        "target_productivity": score[1].reshape(-1),
        "target_growth": np.full(4 * NODE_COUNT, np.nan),
    })


def asof_profitability_history(
        source: pd.DataFrame, provisional: pd.DataFrame,
        prehistory: pd.DataFrame, bounds: pd.DataFrame, nodes: list[str]
) -> np.ndarray:
    """At forecast quarter t, reconstruct each earlier quarter with t's history.

    Completed years: apply proportional Denton to the available annual
    benchmarks through year(t)-1. Keep completed Train/Dev split blocks frozen
    at their original boundaries. The active block is fit to its completed
    annual prefix only, using the preceding frozen block's final ratio.
    This reproduces Stage03's retrospective labels when a split has fully
    completed, but never lets a later year enter an earlier forecast origin.
    Unfinished year y == year(t): use its previous-year annual-GOS/VA
    ratio and only already observed quarterly VA/GO (the provisional bank).
    The output is [80 forecast origins, 12 previous quarters, 59 industries].
    Row t never uses annual GOS for year(t) or any economic observation >= t.
    """
    if len(nodes) != NODE_COUNT or len(set(nodes)) != NODE_COUNT:
        raise ValueError("Expected exact 59-node profitability input history")
    # engineer_target_history() uses alphabetically ordered node columns.
    # Match that exact order: otherwise a numerically correct annual GOS
    # estimate would be assigned to a DIFFERENT industry's feature row.
    nodes = sorted(nodes)
    quarters = pd.period_range(HISTORY_START, MODEL_END, freq="Q").astype(str).tolist()
    va = bea_panel(source, "current_va", "quarterly", quarters, nodes).to_numpy()
    go = bea_panel(source, "current_go", "quarterly", quarters, nodes).to_numpy()
    completed_years = [str(year) for year in range(2005, 2025)]
    annual = bea_panel(source, "gos", "annual", completed_years, nodes).to_numpy()
    if not (go > 0).all():
        raise RuntimeError("Nonpositive nominal output in profitability history")
    reference = bounds.loc[bounds.component.eq("profitability"), ["min", "max"]]
    if reference.empty or reference["min"].nunique() != 1 or reference["max"].nunique() != 1:
        raise RuntimeError("Invalid Train-only profitability normalization")
    low, high = reference.iloc[0].to_numpy(float)
    if not low < high:
        raise RuntimeError("Degenerate profitability normalization")

    # One completed-year snapshot per prediction year's first quarter.
    # Never solve with an annual benchmark dated after that snapshot.
    frozen_blocks: dict[int, np.ndarray] = {}
    completed_snapshots: dict[int, np.ndarray] = {}
    for last_year in range(2005, 2025):
        completed = np.full((len(quarters), NODE_COUNT), np.nan)
        previous_ratio = None
        for start_year, block_end in ((2005, 2019), (2020, 2022), (2023, 2025)):
            if last_year < start_year:
                break
            last_in_block = min(last_year, block_end)
            first = (start_year - 2005) * 4
            stop = (last_in_block - 2005 + 1) * 4
            if start_year in frozen_blocks:
                quarterly = frozen_blocks[start_year]
            else:
                quarterly = np.column_stack([
                    proportional_denton(
                        va[first:stop, j],
                        annual[start_year - 2005:last_in_block - 2005 + 1, j],
                        previous_ratio=(None if previous_ratio is None
                                        else float(previous_ratio[j])))
                    for j in range(NODE_COUNT)])
                if last_in_block == block_end:
                    frozen_blocks[start_year] = quarterly
            completed[first:stop] = quarterly
            previous_ratio = quarterly[-1] / va[stop - 1]
            if last_in_block < block_end:
                break
        completed_snapshots[last_year] = 100. * (completed / go - low) / (high - low)
    # 2005 is already a completed year by the first 2006 prediction.
    scores = pd.concat([
        prehistory[["quarter", "node_key", "target_profitability"]],
        provisional[["quarter", "node_key", "target_profitability"]],
    ], ignore_index=True)
    reference_index = pd.MultiIndex.from_product(
        [quarters, nodes], names=["quarter", "node_key"])
    if (len(scores) != len(reference_index) or scores.duplicated(
            ["quarter", "node_key"]).any()):
        raise RuntimeError("Provisional profitability history has missing/duplicate rows")
    provisional_values = (scores.set_index(["quarter", "node_key"])
                          .reindex(reference_index)["target_profitability"]
                          .to_numpy(float).reshape(len(quarters), NODE_COUNT))
    if not np.isfinite(provisional_values).all():
        raise RuntimeError("Missing historical profitability input")
    if not np.allclose(
            completed_snapshots[2005][:4], provisional_values[:4],
            rtol=1e-12, atol=1e-10):
        raise RuntimeError("2005 annual GOS is not consistently reconstructed")

    view = np.full((80, 12, NODE_COUNT), np.nan)
    years = np.asarray([int(q[:4]) for q in quarters])
    for origin in range(4, len(quarters)):
        forecast_year = years[origin]
        completed = completed_snapshots[forecast_year - 1]
        for depth in range(1, 13):
            past = origin - depth
            if past < 0:
                continue
            value = (completed[past] if years[past] < forecast_year
                     else provisional_values[past])
            view[origin - 4, depth - 1] = value
    if not np.isfinite(view[:, 0]).all() or np.isinf(view).any():
        raise RuntimeError("Incomplete next-quarter past profitability")
    return view


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    node = pd.read_csv(NODE_02, dtype={"quarter": str, "node_key": str}, low_memory=False)
    global_features = pd.read_csv(GLOBAL_02, dtype={"quarter": str}, low_memory=False)
    masks = pd.read_csv(MASK_02, dtype={"quarter": str, "node_key": str}, low_memory=False)
    source_nodes = set(load_nodes().node_key)
    source_quarters = set(pd.period_range(HISTORY_START, MODEL_END, freq="Q").astype(str))
    if (len(node) != len(source_quarters) * NODE_COUNT
            or node.duplicated(["quarter", "node_key"]).any()
            or set(node.node_key) != source_nodes
            or set(node.quarter) != source_quarters
            or len(global_features) != len(source_quarters)
            or set(global_features.quarter) != source_quarters):
        raise RuntimeError("Stage 02 must cover exactly 84 quarters and 59 industry nodes")
    node_columns = [col for col in node if col not in {"quarter", "node_key"}]
    global_columns = [col for col in global_features if col != "quarter"]
    summary = json.loads(SUMMARY_02.read_text(encoding="utf-8"))
    if summary.get("nodes") != NODE_COUNT or summary.get("quarters") != len(source_quarters):
        raise RuntimeError("Stage 02 node/quarter summary disagrees with Stage 01")
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
    """Plot the equal-weight 59-industry quarterly mean of all four targets.

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
    if not counts.eq(NODE_COUNT).all().all():
        raise RuntimeError("Target mean plot requires exactly 59 industries per quarter")
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
        ax.text(0.0, 1.015, "59-industry mean; fixed 2006Q1–2019Q4 Train reference",
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
        "mean_method": "arithmetic mean over all 59 industries, separately for each quarter and scored target",
        "quarters": len(means), "industries_per_quarter": NODE_COUNT,
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
        if (len(part) != len(quarters) * NODE_COUNT
                or part["quarter"].nunique() != len(quarters)
                or not part.groupby("quarter")["node_key"].nunique().eq(NODE_COUNT).all()
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


def engineer_target_history(
        targets: pd.DataFrame, prehistory: pd.DataFrame | None = None,
        profitability_asof: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact 42-feature bank from leakage-free historical score estimates.

    Feature row t uses historical input scores through t-1, never Y_t or
    future inputs, including rolling and peer statistics. Source labels are
    separately preserved in build_targets; history profitability is a proxy.
    Optional 2005 profitability/productivity observations seed early Train
    history; 2005 growth stays missing without 2004 quarterly real GO.
    With profitability_asof, ALL profitability lags and rolling/peer features
    are computed from the same forecast-origin-specific completed-year values.
    """
    keys = ["quarter", "node_key"]
    quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str).tolist()
    nodes = sorted(targets.node_key.unique())
    index = pd.MultiIndex.from_product([quarters, nodes], names=keys)
    if len(nodes) != NODE_COUNT or len(targets) != len(index) or targets.duplicated(keys).any():
        raise ValueError("Require exact 80-quarter/59-industry Target grid")
    aligned = targets.set_index(keys).reindex(index)
    original = aligned[TARGETS].to_numpy(float).reshape(80, NODE_COUNT, 4)
    if not np.isfinite(original).all():
        raise ValueError("Nonfinite Target in historical feature inputs")
    if not np.allclose(original[..., 3], original[..., :3].mean(-1),
                       rtol=1e-12, atol=1e-12):
        raise ValueError("ICI identity changed")
    if profitability_asof is not None:
        profitability_asof = np.asarray(profitability_asof, dtype=float)
        if (prehistory is None or profitability_asof.shape != (80, 12, NODE_COUNT)
                or not np.isfinite(profitability_asof[:, 0]).all()
                or np.isinf(profitability_asof).any()):
            raise ValueError("Profitability lookbacks must be complete, past-only 80x12x59")
    prefix = np.full((4, NODE_COUNT, 3), np.nan)
    if prehistory is not None:
        early_quarters = [f"2005Q{q}" for q in range(1, 5)]
        early_index = pd.MultiIndex.from_product(
            [early_quarters, nodes], names=keys)
        if (len(prehistory) != len(early_index)
                or prehistory.duplicated(keys).any()
                or not early_index.equals(
                    prehistory.set_index(keys).sort_index().index)):
            raise ValueError("2005 history must cover four quarters and all 59 nodes")
        seed = prehistory.set_index(keys).reindex(early_index)
        prefix = seed[[f"target_{a}" for a in TARGET_AXES]].to_numpy(float).reshape(
            4, NODE_COUNT, 3)
        if (not np.isfinite(prefix[..., :2]).all()
                or not np.isnan(prefix[..., 2]).all()):
            raise ValueError("2005 history must include only known profitability/productivity")
    axes = {axis: np.concatenate((prefix[:, :, i], original[:, :, i]), axis=0)
            for i, axis in enumerate(TARGET_AXES)}
    lag_cache = {}
    features = {}
    registry = []

    def lag(axis, distance):
        if distance < 1:
            raise ValueError("Contemporaneous/future Target leakage")
        key = (axis, distance)
        if key not in lag_cache:
            if axis == "profitability" and profitability_asof is not None:
                if distance > profitability_asof.shape[1]:
                    raise ValueError("Insufficient as-of profitability lookback")
                lag_cache[key] = profitability_asof[:, distance - 1].copy()
            else:
                result = np.full((84, NODE_COUNT), np.nan)
                if distance < 84:
                    result[distance:] = axes[axis][:-distance]
                lag_cache[key] = result[4:]
        return lag_cache[key]

    def add(name, value, family, axis, depth, peers=False):
        value = np.asarray(value, dtype=float)
        if name in features or value.shape != (80, NODE_COUNT) or np.isinf(value).any():
            raise ValueError(f"Invalid Target feature: {name}")
        features[name] = value.reshape(-1)
        registry.append(dict(feature=name, family=family, source_target_axes=axis,
                             minimum_history_quarters=depth,
                             latest_input_quarter_offset=-1,
                             cross_industry=peers,
                             causality="STRICT_T_MINUS_ONE_OR_EARLIER",
                             scope="NODE_WITH_PRIOR_PEER_DATA" if peers else "NODE_HISTORY"))

    # Previous-quarter input scores are first for readability, never mandatory.
    # Profitability comes from the past-only proxy; other axes are true lags.
    for axis in TARGET_AXES:
        add(f"history_{axis}", lag(axis, 1), "recent_observation", axis, 1)

    for axis in TARGET_AXES:
        recent = lag(axis, 1)
        for depth in TARGET_LAGS:
            add(f"targethist_{axis}_lag{depth:02d}", lag(axis, depth),
                "historical_level", axis, depth)

        # Change from last observed Target to earlier historical observations.
        for distance in TARGET_CHANGES:
            add(f"targethist_{axis}_change{distance:02d}",
                recent - lag(axis, 1 + distance),
                "historical_momentum", axis, 1 + distance)

        if axis == "profitability" and profitability_asof is not None:
            for window in TARGET_WINDOWS:
                historical = np.stack(
                    [lag(axis, depth) for depth in range(1, window + 1)])
                complete = np.isfinite(historical).all(axis=0)
                observed = np.where(np.isfinite(historical), historical, 0.)
                means = np.where(complete, observed.mean(axis=0), np.nan)
                deviations = np.where(complete, observed.std(axis=0, ddof=0), np.nan)
                add(f"targethist_{axis}_mean_w{window:02d}",
                    means, "historical_mean", axis, window)
                add(f"targethist_{axis}_std_w{window:02d}",
                    deviations, "historical_volatility", axis, window)
        else:
            full_recent = np.full((84, NODE_COUNT), np.nan)
            full_recent[1:] = axes[axis][:-1]
            full_series = pd.DataFrame(
                full_recent, index=[f"2005Q{q}" for q in range(1, 5)] + quarters,
                columns=nodes)
            for window in TARGET_WINDOWS:
                r = full_series.rolling(window, min_periods=window)
                add(f"targethist_{axis}_mean_w{window:02d}",
                    r.mean().to_numpy()[4:], "historical_mean", axis, window)
                add(f"targethist_{axis}_std_w{window:02d}",
                    r.std(ddof=0).to_numpy()[4:], "historical_volatility", axis, window)

        # Four-quarter least-squares slope, using t-4 .. t-1 only.
        window = 4
        history = np.stack([lag(axis, k) for k in range(window, 0, -1)])
        indices = np.arange(window, dtype=float) - (window - 1) / 2.
        add(f"targethist_{axis}_trend_w04",
            np.sum(history * indices[:, None, None], axis=0) / np.sum(indices ** 2),
            "historical_trend", axis, window)

        # Prior-quarter relative rank/gap among all 59 industries.
        series = pd.DataFrame(recent, index=quarters, columns=nodes)
        peer_mean = series.mean(axis=1).to_numpy()[:, None]
        add(f"targethist_{axis}_peer_gap",
            recent - peer_mean, "prior_industry_relative_level", axis, 1, True)
        add(f"targethist_{axis}_peer_rank",
            series.rank(axis=1, pct=True).to_numpy(),
            "prior_industry_relative_rank", axis, 1, True)

    result = pd.concat([index.to_frame(index=False), pd.DataFrame(features)], axis=1)
    feature_registry = pd.DataFrame(registry)
    if len(features) != 42 or result.columns[2:5].tolist() != HISTORY_COLUMNS:
        raise RuntimeError("Expected 42 target features with direct lags first")
    first = result.loc[result.quarter == MODEL_START]
    if prehistory is None:
        if not first[list(features)].isna().all().all():
            raise RuntimeError("Missing 2005 source may not create model history")
    elif (not first[["history_profitability", "history_productivity"]].notna().all().all()
          or not first["history_growth"].isna().all()):
        raise RuntimeError("Only validated 2005 historical observations may be restored")
    if result.duplicated(keys).any() or feature_registry.feature.duplicated().any():
        raise RuntimeError("Duplicated target-feature rows or names")
    return result, feature_registry


def assemble() -> dict:
    require_files()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    registry = load_nodes()
    source = load_latest_sources()

    print("[1/4] Calculating profitability, productivity, growth and ICI", flush=True)
    targets, raw, score_bounds, denton_check, funds_correction = build_targets(source, registry)
    history_inputs = causal_history_targets(
        source, targets, score_bounds, registry.node_key.tolist())
    prior_inputs = pretrain_history_inputs(
        source, score_bounds, registry.node_key.tolist())
    past_profitability = asof_profitability_history(
        source, history_inputs, prior_inputs, score_bounds, registry.node_key.tolist())
    history, target_feature_registry = engineer_target_history(
        history_inputs, prehistory=prior_inputs, profitability_asof=past_profitability)
    target_features = target_feature_registry.feature.tolist()
    print("[2/4] Joining the 115 original Stage 02 features", flush=True)
    node, glob, node_features, global_features = load_features()
    if targets.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError("Duplicate target rows")
    joined = targets.merge(history, on=["quarter", "node_key"], how="left",
                           validate="one_to_one")
    joined = joined.merge(node, on=["quarter", "node_key"], how="left",
                           validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("A target row is missing Stage 02 industry features")
    joined = joined.drop(columns="_merge")
    joined = joined.merge(glob, on="quarter", how="left", validate="many_to_one",
                          indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("A target row is missing Stage 02 global features")
    joined = joined.drop(columns="_merge")
    # All target-only engineered features appear before all 115 original inputs.
    columns = IDS + TARGETS + target_features + node_features + global_features
    joined = joined[columns].sort_values(["quarter", "node_key"], kind="stable").reset_index(drop=True)
    expected_quarters = pd.period_range(MODEL_START, MODEL_END, freq="Q").astype(str)
    if (len(joined) != 80 * NODE_COUNT or joined.node_key.nunique() != NODE_COUNT
            or joined.quarter.nunique() != 80
            or len(joined.columns) != len(IDS) + len(TARGETS) + len(target_features) + 115
            or len(node_features) + len(global_features) != 115
            or not joined.groupby("quarter").node_key.nunique().eq(NODE_COUNT).all()
            or set(joined.quarter) != set(expected_quarters)):
        raise RuntimeError("Incomplete 80-quarter x 59-industry integration")
    components = joined[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(components).all():
        raise RuntimeError("The three axes or ICI contain missing or infinite values")
    if not np.allclose(components[:, 3], components[:, :3].mean(axis=1), atol=1e-10):
        raise RuntimeError("ICI is not the arithmetic mean of the three target scores")
    features = joined[node_features + global_features]
    model_inputs = joined[target_features + node_features + global_features]
    if np.isinf(model_inputs.to_numpy(dtype=float)).any():
        raise RuntimeError("Input features contain infinite values")
    first = joined.loc[joined.quarter == MODEL_START]
    if (not first[["history_profitability", "history_productivity"]].notna().all().all()
            or not first["history_growth"].isna().all()
            or joined.loc[joined.quarter != MODEL_START, target_features[:3]].isna().any().any()):
        raise RuntimeError("Past-only Target features violate their minimum history")

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
                                  "TARGET" if name in TARGETS else
                                  "TARGET_DERIVED_FEATURE" if name in target_features else "FEATURE"),
         "scope": ("NODE" if name in target_features or name in node_features else
                   "GLOBAL" if name in global_features else "ROW")}
        for name in joined.columns
    ]).to_csv(RESULTS / "integrated_column_registry.csv", index=False, encoding="utf-8-sig")
    target_feature_registry.to_csv(
        RESULTS / "target_feature_registry.csv", index=False, encoding="utf-8-sig")
    print("[4/4] Plotting the four targets' quarterly 59-industry means", flush=True)
    target_mean_timeseries = save_target_mean_timeseries(joined)
    report = {
        "status": "PASS", "policy": "LATEST_OFFICIAL_VALUES_ONLY",
        "outputs": {key: path.relative_to(ROOT).as_posix()
                    for key, path in SPLIT_FILES.items()},
        "period": [MODEL_START, MODEL_END], "quarters": 80,
        "industries": NODE_COUNT, "rows": len(joined), "columns": len(joined.columns),
        "split_periods": {key: list(periods)
                          for key, periods in SPLIT_PERIODS.items()},
        "split_rows": {key: len(part) for key, part in splits.items()},
        "split_ratio_quarters": {
            key: part["quarter"].nunique() / len(expected_quarters)
            for key, part in splits.items()
        },
        "identifier_columns": IDS, "target_columns": TARGETS,
        "target_count": len(TARGETS),
        "previous_target_history_columns": HISTORY_COLUMNS,
        "previous_target_history_lag_quarters": 1,
        "previous_target_history_source": (
            "Profitability: completed past years use their OWN annual GOS "
            "in forecast-origin-bounded Denton with frozen Train/Dev blocks; "
            "unfinished current year uses the previous year's annual GOS "
            "and observed quarterly VA/GO. "
            "All 14 profitability history features use the same forecast-origin view; "
            "productivity: original prior-quarter scores from 2006 and "
            "equivalent component-derived 2005 history; "
            "growth: original prior-quarter score when defined, no invented "
            "2005 growth. All sources latest revised official values."
        ),
        "previous_target_history_first_quarter_nan_cells": int(
            first[HISTORY_COLUMNS].isna().sum().sum()),
        "recovered_2005_history_axes": ["profitability", "productivity"],
        "profitability_history_completed_year_policy": (
            "Denton fitted only to annual GOS and quarterly VA from completed "
            "years strictly before the predicted year, freezing each completed "
            "Train/Dev/Test split block at its boundary"
        ),
        "profitability_history_current_year_policy": (
            "Previous year's annual GOS/mean VA times already-observed "
            "current-year quarterly VA divided by nominal gross output"
        ),
        "target_engineered_feature_count": len(target_features),
        "target_engineered_feature_groups": {
            str(k): int(v) for k, v in
            target_feature_registry.family.value_counts().sort_index().items()
        },
        "target_feature_registry": "Results/03_integrated_dataset/target_feature_registry.csv",
        "target_feature_observation_policy": (
            "Every row t uses only information through t-1; "
            "all rolling and peer calculations are strictly lagged; "
            "insufficient early history remains NaN without backfilling"
        ),
        "missing_engineered_target_feature_cells": int(
            joined[target_features].isna().sum().sum()),
        "node_feature_count": len(node_features),
        "global_feature_count": len(global_features),
        "total_feature_count": len(node_features) + len(global_features),
        "model_input_feature_count": len(target_features) + len(node_features) + len(global_features),
        "auto_generated_feature_count": len(target_features),
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
            "fixed Train-only pooled 59-industry Min-Max, "
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
    # JSON saved above remains UTF-8; escape non-ASCII for CP949 terminals.
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


if __name__ == "__main__":
    assemble()
