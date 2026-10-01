#!/usr/bin/env python3
"""Prepare PRISM stage-03 splits without introducing a static future scaler.

Targets/history and external features are kept in their raw point-in-time units.
Only observed-value masks are appended here. Actual normalization is performed
inside stage 05 separately for every sample using observations available through
that sample's forecast origin. This file is therefore a storage/validation stage,
not an authoritative model scaler.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "Data" / "03_integrated_dataset"
OUTPUT_DIR = ROOT / "Data" / "04_preprocessing"
RESULTS_DIR = ROOT / "Results" / "04_preprocessing"
FEATURE_SUMMARY = ROOT / "Results" / "02_initial_feature_pool" / "feature_pool_summary.json"

IDENTIFIERS = ["quarter", "node_key", "node_name"]
TARGETS = [
    "target_profitability",
    "target_productivity",
    "target_growth",
    "target_ici",
]
TARGET_RAW = [
    "target_profitability_raw",
    "target_productivity_raw",
    "target_growth_raw",
]
HISTORY = [
    "history_profitability",
    "history_productivity",
    "history_growth",
    "history_ici",
]
HISTORY_RAW = [
    "history_profitability_raw",
    "history_productivity_raw",
    "history_growth_raw",
]

INPUTS = {
    "train": INPUT_DIR / "raw_train.csv",
    "val": INPUT_DIR / "raw_val.csv",
    "test": INPUT_DIR / "raw_test.csv",
}
OUTPUTS = {
    "train": OUTPUT_DIR / "preprocessed_train.csv",
    "val": OUTPUT_DIR / "preprocessed_val.csv",
    "test": OUTPUT_DIR / "preprocessed_test.csv",
}
LEGACY_OUTPUTS = [OUTPUT_DIR / "preprocessed_holdout_2025.csv"]


def transform_split(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    values = df[feature_cols].apply(pd.to_numeric, errors="raise")
    observed = values.notna()

    masks = observed.astype(np.int8)
    masks.columns = [f"mask__{col}" for col in feature_cols]

    return pd.concat(
        [
            df[IDENTIFIERS + TARGETS + TARGET_RAW + HISTORY + HISTORY_RAW].reset_index(drop=True),
            values.reset_index(drop=True),
            masks.reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in LEGACY_OUTPUTS:
        if path.exists():
            path.unlink()

    train = pd.read_csv(INPUTS["train"], dtype={"quarter": str, "node_key": str, "node_name": str})
    feature_cols = [
        c for c in train.columns
        if c not in IDENTIFIERS + TARGETS + TARGET_RAW + HISTORY + HISTORY_RAW
    ]
    feature_summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    expected_features = int(feature_summary["total_initial_model_channels"])
    if len(feature_cols) != expected_features:
        raise RuntimeError(f"Expected {expected_features} feature columns, found {len(feature_cols)}")
    if train[HISTORY + HISTORY_RAW].isna().any().any():
        raise RuntimeError("Causal history contains missing values")
    if not np.isfinite(train[HISTORY + HISTORY_RAW].to_numpy(dtype=float)).all():
        raise RuntimeError("Causal history contains non-finite values")

    expected_columns = list(train.columns)
    split_rows: dict[str, int] = {}
    output_columns: int | None = None
    for split in INPUTS:
        df = train if split == "train" else pd.read_csv(
            INPUTS[split], dtype={"quarter": str, "node_key": str, "node_name": str}
        )
        if list(df.columns) != expected_columns:
            raise RuntimeError(f"{split}: columns differ from train")

        out = transform_split(df, feature_cols)
        finite_required = out[HISTORY + HISTORY_RAW + TARGETS + TARGET_RAW].to_numpy(dtype=float)
        if not np.isfinite(finite_required).all():
            raise RuntimeError(f"{split}: non-finite target/history value after preprocessing")
        out.to_csv(OUTPUTS[split], index=False, encoding="utf-8-sig")
        split_rows[split] = len(out)
        output_columns = len(out.columns)
        print(f"{split}: rows={len(out)} cols={len(out.columns)} -> {OUTPUTS[split].relative_to(ROOT)}")

    summary = {
        "outputs": {split: str(path.relative_to(ROOT)) for split, path in OUTPUTS.items()},
        "split_rows": split_rows,
        "causal_history_columns": HISTORY,
        "causal_history_count": len(HISTORY),
        "causal_history_raw_columns": HISTORY_RAW,
        "target_raw_columns": TARGET_RAW,
        "feature_count": len(feature_cols),
        "mask_count": len(feature_cols),
        "output_columns": output_columns,
        "feature_scaling": "NONE in stage 04; stage 05 applies sample-origin expanding normalization",
        "missing_value_rule": "raw missing feature values preserved as NaN with observed-value mask",
        "targets_changed": False,
        "status": "PASS",
    }
    (RESULTS_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("PASS")


if __name__ == "__main__":
    main()
