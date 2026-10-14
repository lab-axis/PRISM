#!/usr/bin/env python3
"""PRISM Stage08: Train-only B-MTGNN ablation ranking and exhaustive ranked Top-K.

Frozen parent: Stage06/07 Structural B-MTGNN, L=1, 10 paired seeds,
36 common 2011-2019 Train origins, 75 epochs, selected Stage06 preprocessing.
Stage07 selected no removal: all 157 main-input features are available.

1. Load the ACTUAL Stage06 seed checkpoints. Rank 157 main-input features
   using signed Train IN-SAMPLE change in three-axis MAE after individually
   zeroing both normalized values and missing indicators. A frozen model
   is evaluated with dropout OFF for attribution; deployment uses 30-draw
   MC dropout. Also record seed and three-epoch Train-block sensitivity.
2. Freeze that single ranking. Fit ALL 156 smaller ranked prefixes K=1..156
   with Stage07's exact compact K-input training function, initialization,
   fixed structural prior, context, anchor and MC inference; reuse the ten
   Stage06 K=157 forecasts without retraining. No coarse/fine shortcut.
3. Compute original-unit Dev errors for EACH of ten matched seed forecasts,
   then average the seed metrics. Only adopt a smaller K if mean three-axis
   MAE improves, mean ICI MAE does not worsen, and >=2 Dev years improve.
4. At a predeclared fixed K=79, train ten independent random subsets with
   exactly the same ten fitting seeds. They are descriptive controls,
   evaluated regardless of whether any compact K is adopted.

Optimality means the best eligible prefix of ONE fixed ranking on the 157-K
grid, NEVER the best of 2^157 subsets. All Dev conclusions are conditional:
Stages04-08 repeatedly use the 2020-22 Dev period. Train attribution is
in-sample, not causal, held-out or statistical evidence of importance.
The future Test split is not opened here. No legacy Stage08 compatibility.

    py -m pytest -q Tests/test_08_feature_selection.py
    py Sources/08_feature_selection.py --dry-run
    py -u Sources/08_feature_selection.py --ranking-only
    py -u Sources/08_feature_selection.py --run
    py Sources/08_feature_selection.py --status
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/08_feature_selection.py"
SOURCE07 = ROOT / "Sources/07_correlated_feature_removal.py"
RESULTS = ROOT / "Results/08_feature_selection"
RESULT07 = ROOT / "Results/07_correlated_feature_removal"

FEATURE_COUNT = 157
SEEDS = tuple(range(2026, 2036))
TRAIN_ORIGINS = tuple(range(20, 56))
TRAIN_BLOCKS = ((20, 32), (32, 44), (44, 56))  # 2011-13, 2014-16, 2017-19
ALL_K = tuple(range(1, FEATURE_COUNT + 1))
RANDOM_K = 79
RANDOM_SUBSET_SEEDS = tuple(range(20261013, 20261023))
EPS = 1e-9
KEYS = ("axis_mae", "ici_mae", "profitability_mae",
        "productivity_mae", "growth_mae", "2020_mae",
        "2021_mae", "2022_mae", "2020_ici_mae",
        "2021_ici_mae", "2022_ici_mae")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def stage07():
    spec = importlib.util.spec_from_file_location("_prism08_stage07", SOURCE07)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def upstream() -> dict:
    m07 = stage07()
    ctx = m07.upstream()
    lock = read_json(RESULT07 / "selected_features.json")
    manifest = read_json(RESULT07 / "run_manifest.json")
    baseline, individual = ctx["m06"].seed_policy_metrics(
        ctx["original_forecasts"], ctx["panel"])
    if (lock.get("status") != "LOCKED"
            or manifest.get("status") != "PASS"
            or lock.get("selected_policy") != "no_removal"
            or lock.get("selected_mode") != "original"
            or lock.get("selected_indices") != list(range(FEATURE_COUNT))
            or lock.get("selected_feature_count") != FEATURE_COUNT
            or lock.get("gnn_models") != ["B-MTGNN"]
            or lock.get("seeds") != list(SEEDS)
            or lock.get("lookback_quarters") != 1
            or lock.get("epochs") != 75
            or lock.get("stage07_signature") != manifest.get("signature")
            or manifest.get("signature") != m07.run_signature(m07.run_contract(ctx))
            or manifest["contract"]["source_sha256"] != digest(SOURCE07)
            or ctx["full_fold"]["train_x"].shape != (36, 1, 59, 157)
            or ctx["full_fold"]["val_x"].shape != (12, 1, 59, 157)
            or not np.array_equal(ctx["full_fold"]["train_origins"], TRAIN_ORIGINS)
            or ctx["epochs"] != 75
            or lock.get("test_opened") is not False):
        raise RuntimeError("Stage08 requires the exact locked Stage07 no-removal L1 contract")
    for key in KEYS:
        if abs(baseline[key] - lock["final_dev"][key]) > 1e-10:
            raise RuntimeError("Stage07 individual-seed baseline mismatch")
    ctx.update(m07=m07, stage07_lock=lock, baseline=baseline,
               baseline_individual=individual,
               stage07_lock_hash=digest(RESULT07 / "selected_features.json"),
               stage07_manifest_hash=digest(RESULT07 / "run_manifest.json"))
    return ctx


def contract(ctx: dict) -> dict:
    return dict(
        stage="08_SINGLE_CANONICAL_BMTGNN_EXHAUSTIVE_RANKED_K",
        source_sha256=digest(SOURCE),
        stage07_source_sha256=digest(SOURCE07),
        stage07_selection_sha256=ctx["stage07_lock_hash"],
        stage07_manifest_sha256=ctx["stage07_manifest_hash"],
        stage06_selection_sha256=ctx["upstream06_lock_sha256"],
        stage06_manifest_sha256=ctx["upstream06_manifest_sha256"],
        original_train_dev_hashes=ctx["panel_hashes"],
        stage06_model_input_sha256=ctx["upstream06_input_sha256"],
        model="B-MTGNN", structural=True, lookback=1,
        core_lookback=20, train_origins=list(TRAIN_ORIGINS),
        epochs=75, seeds=list(SEEDS), dropout_draws=30,
        trained_on="2011Q1-2019Q4 common 36 Train origins",
        ranking_on="TRAIN_IN_SAMPLE_36_ORIGINS_DROPOUT_OFF",
        ranking_blocks=[list(x) for x in TRAIN_BLOCKS],
        ranking="SIGNED_MAE_INCREASE_ZERO_INPUT_AND_MISSING_MASK",
        ranked_prefixes=list(ALL_K), globally_optimal_subset=False,
        per_seed_metric_then_mean=True,
        safeguards="MAE improve AND ICI nonworse AND >=2 of 3 Dev years",
        random_control_k=RANDOM_K,
        random_subset_seeds=list(RANDOM_SUBSET_SEEDS),
        random_controls_independent_of_dev_adoption=True,
        original_prior_context_anchor_and_internal_graph_preserved=True,
        repeated_dev_selection=True, independent_test=False,
        test_opened=False)


def signature(spec: dict) -> str:
    return hashlib.sha256(json.dumps(
        spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def subset(ctx: dict, indices: tuple[int, ...], label: str):
    if (not indices or tuple(sorted(set(indices))) != indices
            or indices[0] < 0 or indices[-1] >= FEATURE_COUNT):
        raise ValueError("Expected sorted unique Stage08 original feature indices")
    kept = set(indices)
    return ctx["m07"].Subset(
        label, None, indices,
        tuple(i for i in range(FEATURE_COUNT) if i not in kept),
        pd.DataFrame(), 0)


def load_frozen_seed(ctx: dict, seed: int, device):
    """Restore an actual Stage06-selected 75-epoch checkpoint, never Stage04."""
    if seed not in SEEDS:
        raise ValueError("Unknown frozen Stage06 seed")
    root = ctx["m06"].RESULTS / "jobs" / ctx["locked_policy"].id
    stem = f"B-MTGNN__s{seed}"
    meta = read_json(root / (stem + ".json"))
    path = root / (stem + ".pt")
    stage06_manifest = read_json(ctx["m06"].RESULTS / "run_manifest.json")
    if (meta.get("signature") != stage06_manifest["signature"]
            or meta.get("policy") != ctx["locked_policy"].id
            or meta.get("seed") != seed
            or meta.get("model") != "B-MTGNN"
            or meta.get("epoch") != 75
            or meta.get("input_sha256") != ctx["upstream06_input_sha256"]
            or meta.get("checkpoint_sha256") != digest(path)):
        raise RuntimeError("Stage06 frozen checkpoint does not match selected policy")
    frozen = torch.load(path, map_location="cpu", weights_only=True)
    if (frozen.get("signature") != meta["signature"]
            or frozen.get("policy") != meta["policy"]
            or frozen.get("model") != "B-MTGNN"
            or frozen.get("seed") != seed
            or frozen.get("epoch") != 75
            or frozen.get("input_sha256") != meta["input_sha256"]):
        raise RuntimeError("Stage06 frozen checkpoint contents drifted")
    original = subset(ctx, tuple(range(FEATURE_COUNT)), "no_removal")
    net = ctx["m07"].instantiate_subset(ctx, original, seed, compact=False)
    net.load_state_dict(frozen["model_state_dict"], strict=True)
    return net.to(device).eval()


def ranking_tensors(ctx: dict, device):
    fold = ctx["full_fold"]
    if (not np.array_equal(fold["train_origins"], TRAIN_ORIGINS)
            or tuple(ctx["panel"].quarters[t] for t in
                     (20, 31, 32, 43, 44, 55)) !=
            ("2011Q1", "2013Q4", "2014Q1", "2016Q4", "2017Q1", "2019Q4")):
        raise RuntimeError("Train-only attribution period changed")
    z = {key: torch.as_tensor(fold["train_" + key], device=device)
         for key in ("x", "mask", "last", "structure", "context")}
    truth = ctx["panel"].y[np.asarray(TRAIN_ORIGINS)]
    return z, truth


def seed_importance(net, z: dict, truth: np.ndarray,
                    center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, float]:
    """Signed change in original-unit MAE, across all Train and 3 blocks."""
    if (z["x"].shape != (36, 1, 59, 157)
            or truth.shape != (36, 59, 3)
            or net.training):
        raise ValueError("Train ranking requires 36 frozen L1 observations")
    graph = torch.eye(59, device=z["x"].device)
    x, mask = z["x"].clone(), z["mask"].clone()

    def errors() -> np.ndarray:
        with torch.inference_mode():
            out = net(x, mask, graph, z["last"],
                      z["structure"], z["context"])[:, 0]
        prediction = out.cpu().numpy().astype(np.float64) * scale + center
        return np.abs(prediction - truth).mean(axis=(1, 2))

    reference = errors()
    effects = np.empty((FEATURE_COUNT, 4), np.float64)
    for j in range(FEATURE_COUNT):
        original_x, original_mask = x[..., j].clone(), mask[..., j].clone()
        x[..., j], mask[..., j] = 0., 0.
        delta = errors() - reference
        effects[j] = (delta.mean(),
                      delta[:12].mean(), delta[12:24].mean(), delta[24:].mean())
        x[..., j], mask[..., j] = original_x, original_mask
    if not np.isfinite(effects).all():
        raise RuntimeError("Nonfinite Train-only ablation")
    return effects, float(reference.mean())


def ranking_order(scores: np.ndarray) -> tuple[int, ...]:
    if scores.shape != (FEATURE_COUNT,) or not np.isfinite(scores).all():
        raise ValueError("Importance must cover all 157 predictors")
    return tuple(sorted(range(FEATURE_COUNT), key=lambda i: (-scores[i], i)))


def rank_stability(samples: np.ndarray) -> dict:
    """Seed and Train-block rank concordance; descriptive, not a p-value."""
    if samples.shape != (10, FEATURE_COUNT, 4):
        raise ValueError("Ten seeds, 157 features and 3 temporal blocks required")
    overall = samples[:, :, 0]
    mean = overall.mean(axis=0)
    orders = [ranking_order(values) for values in overall]
    block_orders = [
        ranking_order(samples[:, :, i].mean(axis=0)) for i in (1, 2, 3)]

    def rank_positions(order):
        ranks = np.empty(FEATURE_COUNT)
        ranks[list(order)] = np.arange(FEATURE_COUNT)
        return ranks

    def concordance(orders):
        ranks = np.stack([rank_positions(order) for order in orders])
        correlations = np.corrcoef(ranks)
        return float(correlations[np.triu_indices(len(orders), 1)].mean())

    chosen = ranking_order(mean)
    topk_overlap = {}
    for k in (40, 79, 120):
        reference = set(chosen[:k])
        topk_overlap[str(k)] = float(np.mean([
            len(reference & set(order[:k])) /
            len(reference | set(order[:k])) for order in orders]))
    return dict(
        ten_seed_mean_pairwise_rank_spearman=concordance(orders),
        three_train_block_mean_pairwise_rank_spearman=concordance(block_orders),
        mean_topk_jaccard_vs_ten_seed_rankings=topk_overlap,
        periods=["2011-2013", "2014-2016", "2017-2019"],
        interpretation="Train in-sample stability; not independent validation")


def validate_ranking(frame: pd.DataFrame, features: list[str]) -> tuple[int, ...]:
    if (len(frame) != FEATURE_COUNT
            or frame.feature_rank.tolist() != list(range(1, FEATURE_COUNT + 1))
            or sorted(frame.original_index.tolist()) != list(range(FEATURE_COUNT))
            or not np.isfinite(frame.mean_train_ablation_delta).all()
            or any(features[int(i)] != name for i, name in
                   zip(frame.original_index, frame.feature))):
        raise RuntimeError("Stage08 ranking must uniquely cover original 157 features")
    order = tuple(int(i) for i in frame.original_index)
    if order != ranking_order(np.asarray(
            frame.set_index("original_index")
            .loc[range(FEATURE_COUNT), "mean_train_ablation_delta"])):
        raise RuntimeError("Ranking no longer matches signed importance")
    return order


def build_ranking(ctx, root: Path, study_sig: str, device):
    ranking_dir = root / "ranking"
    finished = ranking_dir / "feature_importance.csv"
    record = ranking_dir / "provenance.json"
    if finished.exists():
        meta = read_json(record)
        if (meta.get("signature") != study_sig
                or meta.get("csv_sha256") != digest(finished)):
            raise RuntimeError("Old Stage08 ranking cache; no compatibility path")
        frame = pd.read_csv(finished)
        validate_ranking(frame, ctx["panel"].features)
        cached = []
        for seed in SEEDS:
            path = ranking_dir / "seeds" / f"s{seed}.npz"
            proof = read_json(ranking_dir / "seeds" / f"s{seed}.json")
            frozen_dir = ctx["m06"].RESULTS / "jobs" / ctx["locked_policy"].id
            name = f"B-MTGNN__s{seed}"
            source_meta = read_json(frozen_dir / (name + ".json"))
            if (proof.get("signature") != study_sig
                    or proof.get("seed") != seed
                    or proof.get("npz_sha256") != digest(path)
                    or source_meta.get("checkpoint_sha256") != digest(
                        frozen_dir / (name + ".pt"))):
                raise RuntimeError("Frozen checkpoint or ranking seed cache drifted")
            with np.load(path, allow_pickle=False) as stored:
                cached.append(stored["effects"].copy())
        cached = np.stack(cached)
        if cached.shape != (10, FEATURE_COUNT, 4):
            raise RuntimeError("Saved Train ranking is incomplete")
        means = cached[:, :, 0].mean(axis=0)
        saved_means = frame.sort_values("original_index").mean_train_ablation_delta
        if not np.allclose(means, saved_means, rtol=0, atol=1e-12):
            raise RuntimeError("Saved feature ranking does not reproduce seed results")
        stability = rank_stability(cached)
        if stability != meta["stability"]:
            raise RuntimeError("Saved ranking stability does not reproduce seed results")
        return frame, stability
    z, truth = ranking_tensors(ctx, device)
    scale = ctx["full_fold"]["y_scale"]
    center = ctx["full_fold"]["y_center"]
    results = []
    for seed in SEEDS:
        parent = ranking_dir / "seeds"
        path = parent / f"s{seed}.npz"
        meta_path = parent / f"s{seed}.json"
        if path.exists() and meta_path.exists():
            meta = read_json(meta_path)
            if (meta.get("signature") != study_sig
                    or meta.get("seed") != seed
                    or meta.get("npz_sha256") != digest(path)):
                raise RuntimeError("Invalid Stage08 per-seed importance cache")
            with np.load(path, allow_pickle=False) as saved:
                effects = saved["effects"].copy()
        elif path.exists() or meta_path.exists():
            raise RuntimeError("Incomplete Stage08 Train ranking cache")
        else:
            model = load_frozen_seed(ctx, seed, device)
            effects, reference = seed_importance(model, z, truth, center, scale)
            del model
            write_npz(path, effects=effects, reference=np.asarray(reference))
            write_json(meta_path, dict(
                signature=study_sig, seed=seed, npz_sha256=digest(path),
                train_origins=list(TRAIN_ORIGINS), dropout="EVAL_OFF",
                train_in_sample=True, test_opened=False))
        if effects.shape != (FEATURE_COUNT, 4) or not np.isfinite(effects).all():
            raise RuntimeError("Corrupt Stage08 Train feature importance")
        results.append(effects)
        print("08_RANK_SEED", seed, "COMPLETE", flush=True)
    samples = np.stack(results)
    mean = samples.mean(axis=0)
    stability = rank_stability(samples)
    names = ctx["panel"].features
    items = []
    for i in ranking_order(mean[:, 0]):
        items.append(dict(
            feature=names[i], original_index=i,
            feature_role=("target_history" if i < 42 else
                          "industry_source" if i < 66 else "global_source"),
            mean_train_ablation_delta=float(mean[i, 0]),
            sd_across_ten_seeds=float(samples[:, i, 0].std(ddof=1)),
            positive_seed_fraction=float((samples[:, i, 0] > 0).mean()),
            train_2011_2013_delta=float(mean[i, 1]),
            train_2014_2016_delta=float(mean[i, 2]),
            train_2017_2019_delta=float(mean[i, 3])))
    frame = pd.DataFrame(items)
    frame.insert(0, "feature_rank", range(1, FEATURE_COUNT + 1))
    validate_ranking(frame, names)
    ranking_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(finished, index=False, encoding="utf-8-sig")
    write_json(record, dict(
        signature=study_sig, csv_sha256=digest(finished),
        model="B-MTGNN", seeds=list(SEEDS), train_origins=list(TRAIN_ORIGINS),
        no_dev_labels_used_in_ranking=True, train_in_sample=True,
        dropout_off_only_for_ranking=True, stability=stability,
        test_opened=False))
    return frame, stability


def topk_indices(order: tuple[int, ...], k: int) -> tuple[int, ...]:
    if (len(order) != FEATURE_COUNT
            or set(order) != set(range(FEATURE_COUNT))
            or not 1 <= k <= FEATURE_COUNT):
        raise ValueError("Top-K requires complete frozen ranking")
    return tuple(sorted(order[:k]))


def random_indices(seed: int) -> tuple[int, ...]:
    if seed not in RANDOM_SUBSET_SEEDS:
        raise ValueError("Random comparator must use a predeclared subset seed")
    return tuple(sorted(map(int, np.random.default_rng(seed).choice(
        FEATURE_COUNT, RANDOM_K, replace=False))))


def qualification(trial: dict, baseline: dict) -> dict:
    wins = sum(trial[f"{year}_mae"] < baseline[f"{year}_mae"] - EPS
               for year in (2020, 2021, 2022))
    improved = trial["axis_mae"] < baseline["axis_mae"] - EPS
    ici_safe = trial["ici_mae"] <= baseline["ici_mae"] + EPS
    return dict(
        mae_improved=bool(improved),
        ici_nonworsening=bool(ici_safe),
        dev_years_improved=int(wins),
        qualified=bool(improved and ici_safe and wins >= 2))


def select_k(records: list[dict]) -> int:
    """Exhaustive only over prefixes of one fixed ranking, never all subsets."""
    if (len(records) != FEATURE_COUNT
            or sorted(int(r["K"]) for r in records) != list(ALL_K)):
        raise ValueError("Stage08 selection requires all 157 ranked prefixes")
    valid = [r for r in records if r["K"] < FEATURE_COUNT and r["qualified"]]
    if not valid:
        return FEATURE_COUNT
    return int(min(valid, key=lambda r: (r["axis_mae"], r["K"]))["K"])


def score_policy(ctx, root: Path, indices: tuple[int, ...],
                 label: str, kind: str, study_sig: str, device):
    """Matched ten-seed FULL training, with seed-locked 07 compact API."""
    if kind not in ("TOPK", "RANDOM") or not label:
        raise ValueError("Unknown Stage08 policy type")
    current = subset(ctx, indices, label)
    if not current.dropped:
        if kind != "TOPK":
            raise ValueError("Random control cannot be the full baseline")
        metrics = dict(ctx["baseline"])
        individual = ctx["baseline_individual"]
        data_hash = ctx["upstream06_input_sha256"]
    else:
        changed = ctx["m07"].adapt_inputs(ctx["full_fold"], current, "compact")
        data_hash = ctx["m07"].dataset_sha(changed)
        forecasts = []
        for seed in SEEDS:
            folder = root / "models" / label
            stem = f"B-MTGNN__s{seed}"
            meta_path = folder / (stem + ".json")
            prediction_path = folder / (stem + ".npz")
            weights_path = folder / (stem + ".pt")
            if all(p.is_file() for p in (meta_path, prediction_path, weights_path)):
                meta = read_json(meta_path)
                if (meta.get("signature") != study_sig
                        or meta.get("kind") != kind
                        or meta.get("policy") != label
                        or meta.get("seed") != seed
                        or meta.get("original_indices") != list(indices)
                        or meta.get("input_sha256") != data_hash
                        or meta.get("epoch") != 75
                        or meta.get("checkpoint_sha256") != digest(weights_path)
                        or meta.get("prediction_sha256") != digest(prediction_path)):
                    raise RuntimeError("Invalid Stage08 ten-seed model cache")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    prediction = saved["prediction"].copy()
            elif any(p.exists() for p in (meta_path, prediction_path, weights_path)):
                raise RuntimeError("Incomplete Stage08 seed job; refusing mixed cache")
            else:
                fitted = ctx["m07"].fit_subset_model(
                    ctx, current, changed, seed, device)
                prediction = fitted["prediction"]
                folder.mkdir(parents=True, exist_ok=True)
                checkpoint = dict(
                    signature=study_sig, policy=label, kind=kind, seed=seed,
                    input_sha256=data_hash, original_indices=list(indices),
                    feature_count=len(indices), epoch=75,
                    parameters=fitted["parameters"],
                    state_dict=fitted["state_dict"])
                tmp = weights_path.with_suffix(".pt.writing")
                torch.save(checkpoint, tmp)
                os.replace(tmp, weights_path)
                write_npz(prediction_path, prediction=prediction)
                write_json(meta_path, dict(
                    signature=study_sig, policy=label, kind=kind,
                    original_indices=list(indices), seed=seed,
                    input_sha256=data_hash, epoch=75,
                    parameters=fitted["parameters"],
                    checkpoint_sha256=digest(weights_path),
                    prediction_sha256=digest(prediction_path),
                    fit_seconds=fitted["seconds"],
                    inference_draws=fitted["inference_draws"],
                    inference_regime=fitted["inference_regime"],
                    test_opened=False))
                print("08_FIT", label, "K", len(indices),
                      "SEED", seed, flush=True)
            if prediction.shape != (12, 59, 3) or not np.isfinite(prediction).all():
                raise RuntimeError("Invalid Stage08 seed prediction")
            forecasts.append(prediction)
        metrics, individual = ctx["m06"].seed_policy_metrics(
            np.stack(forecasts), ctx["panel"])
    result = dict(
        policy=label, kind=kind, K=len(indices),
        original_indices=list(indices),
        parameters=(ctx["baseline_parameters"]
                    - 2 * ctx["m04"].STAGE04_ACTIVE_WIDTH
                    * (FEATURE_COUNT - len(indices))),
        seed_count=len(SEEDS),
        model="B-MTGNN", lookback=1, epoch=75,
        signature=study_sig, input_sha256=data_hash,
        reused_stage06_baseline=not current.dropped,
        seed_metrics=individual,
        **metrics,
        **(qualification(metrics, ctx["baseline"]) if current.dropped else
           dict(mae_improved=False, ici_nonworsening=True,
                dev_years_improved=0, qualified=False)))
    location = root / "policies" / (label + ".json")
    if location.exists():
        saved = read_json(location)
        if (saved["signature"] != study_sig
                or saved["original_indices"] != list(indices)
                or saved["kind"] != kind
                or abs(saved["axis_mae"] - result["axis_mae"]) > 1e-10
                or abs(saved["ici_mae"] - result["ici_mae"]) > 1e-10):
            raise RuntimeError("Stage08 policy cache does not reproduce seed metrics")
    else:
        write_json(location, result)
    return result


def compare_stage07(rows: list[dict], root: Path):
    previous = pd.read_csv(RESULT07 / "compact_confirmation.csv")
    indexed = {r["K"]: r for r in rows}
    records = []
    for r in previous.itertuples(index=False):
        top = indexed[int(r.kept_count)]
        records.append(dict(
            K=int(r.kept_count), correlation_threshold=r.policy,
            correlation_mae=float(r.axis_mae),
            ranked_topk_mae=float(top["axis_mae"]),
            ranked_minus_correlation_mae=float(
                top["axis_mae"] - r.axis_mae),
            correlation_ici=float(r.ici_mae),
            ranked_topk_ici=float(top["ici_mae"]),
            note="Both share 10 fitting seeds; Dev reused for both policies"))
    pd.DataFrame(records).to_csv(
        root / "stage07_same_k_comparison.csv",
        index=False, encoding="utf-8-sig")


def pareto_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([{key: r[key] for key in
                           ("K", "axis_mae", "ici_mae", "parameters")}
                          for r in rows])
    frame["nondominated"] = [
        not any(
            q.K <= p.K and q.axis_mae <= p.axis_mae
            and q.ici_mae <= p.ici_mae
            and (q.K < p.K or q.axis_mae < p.axis_mae
                 or q.ici_mae < p.ici_mae)
            for q in frame.itertuples(index=False))
        for p in frame.itertuples(index=False)]
    return frame


def generate_plots(root: Path, ranking: pd.DataFrame,
                   scores: pd.DataFrame, chosen: int,
                   controls: list[dict], stability: dict) -> list[str]:
    folder = root / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(fig, name):
        fig.tight_layout()
        path = folder / (name + ".png")
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        figures.append("plots/" + path.name)

    fig, ax = plt.subplots(figsize=(11, 9))
    top = ranking.head(30).iloc[::-1]
    ax.barh(top.feature, top.mean_train_ablation_delta)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Signed Train in-sample MAE change on main-input ablation",
           title="Top-30 input attributions (not causal / not out-of-sample)")
    save(fig, "08_01_train_attribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranking.feature_rank, ranking.mean_train_ablation_delta,
            label="All Train origins")
    for period, col in (("2011-13", "train_2011_2013_delta"),
                        ("2014-16", "train_2014_2016_delta"),
                        ("2017-19", "train_2017_2019_delta")):
        ax.plot(ranking.feature_rank, ranking[col],
                alpha=.6, linewidth=1, label=period)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Rank on complete Train set",
           ylabel="In-sample signed MAE change",
           title="Descriptive temporal sensitivity of frozen-model ranking")
    ax.legend()
    save(fig, "08_02_ranking_period_sensitivity")

    scores = scores.sort_values("K")
    for measure, name in (("axis_mae", "08_03_ranked_k_mae"),
                          ("ici_mae", "08_04_ranked_k_ici")):
        fig, ax = plt.subplots(figsize=(10.5, 5))
        ax.plot(scores.K, scores[measure], linewidth=1.2)
        ax.axhline(float(scores.loc[scores.K.eq(157), measure].iloc[0]),
                   linestyle="--", color="gray", label="Stage07 full 157")
        ax.axvline(chosen, linestyle=":", color="black",
                   label=f"Adopted K={chosen}")
        ax.set(xlabel="K: all 157 ranked prefixes evaluated",
               ylabel=measure + " (ten individual-seed MAEs averaged)",
               title="Conditional 2020-22 development comparison")
        ax.legend()
        save(fig, name)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(scores.axis_mae, scores.ici_mae,
               c=scores.K, s=20, cmap="viridis")
    ax.scatter(
        scores.loc[scores.K.eq(chosen), "axis_mae"],
        scores.loc[scores.K.eq(chosen), "ici_mae"],
        marker="x", s=110, color="black", label="Adopted policy")
    ax.set(xlabel="Mean 3-axis MAE", ylabel="Mean ICI MAE",
           title="Ranked-prefix trade-off; color encodes K")
    ax.legend()
    save(fig, "08_05_ranked_k_pareto")

    fig, ax = plt.subplots(figsize=(10, 5))
    values = [r["axis_mae"] for r in controls]
    ax.scatter(range(len(values)), values, label="Fixed-K random subsets")
    ax.axhline(float(scores.loc[scores.K.eq(RANDOM_K), "axis_mae"].iloc[0]),
               color="black", label=f"Ranked K={RANDOM_K}")
    ax.set(xlabel="Predeclared random subset index",
           ylabel="Ten-seed mean Dev 3-axis MAE",
           title="Descriptive rank vs random at fixed K, no significance claim")
    ax.legend()
    save(fig, "08_06_fixed_k_random_controls")
    return figures


def write_selection(ctx, root: Path, rank: pd.DataFrame,
                    rows: list[dict], chosen: int, controls: list[dict],
                    stability: dict, study_sig: str, plots: list[str]):
    order = validate_ranking(rank, ctx["panel"].features)
    indices = topk_indices(order, chosen)
    ranked = order[:chosen]
    names = ctx["panel"].features
    selected = [names[i] for i in indices]
    removed = [names[i] for i in range(FEATURE_COUNT) if i not in indices]
    baseline = ctx["baseline"]
    by_k = {row["K"]: row for row in rows}
    final = by_k[chosen]
    if (len(by_k) != FEATURE_COUNT or
            final["qualified"] is not (chosen < FEATURE_COUNT)):
        raise RuntimeError("Stage08 final policy violates qualification rule")
    pd.DataFrame([dict(original_index=i, feature=names[i],
                       rank=order.index(i) + 1) for i in indices]).to_csv(
        root / "selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(original_index=i, feature=names[i])
                  for i in range(FEATURE_COUNT) if i not in indices],
                 columns=["original_index", "feature"]).to_csv(
        root / "removed_features.csv", index=False, encoding="utf-8-sig")
    write_npz(root / "selected_feature_indices.npz",
              original_indices=np.asarray(indices, dtype=np.int32),
              ranked_indices=np.asarray(ranked, dtype=np.int32))
    selected_policy = dict(
        stage=8, status="LOCKED",
        selected_K=chosen,
        selected_original_indices=list(indices),
        selected_ranked_indices=list(ranked),
        selected_names=selected,
        removed_names=removed,
        baseline_dev={key: baseline[key] for key in KEYS},
        selected_dev={key: final[key] for key in KEYS},
        selected_qualification={key: final[key] for key in (
            "mae_improved", "ici_nonworsening",
            "dev_years_improved", "qualified")},
        baseline_parameters=ctx["baseline_parameters"],
        selected_parameters=final["parameters"],
        selected_training_reused_stage06=chosen == FEATURE_COUNT,
        all_ranked_prefixes_exhausted=True,
        best_prefix_not_best_arbitrary_subset=True,
        train_attribution_in_sample=True,
        attribution_not_causal=True,
        ten_individual_seed_maes_then_mean=True,
        independent_test=False,
        dev_reused_after_stages_04_05_06_07=True,
        main_input_only_context_prior_anchor_graph_unchanged=True,
        dropout_off_only_for_attribution=True,
        deployed_mc_dropout_draws=30,
        random_controls_at_fixed_k=RANDOM_K,
        random_control_subsets=len(controls),
        rank_stability=stability,
        stage07_lock_sha256=ctx["stage07_lock_hash"],
        stage08_signature=study_sig,
        plots=plots, test_opened=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / "selected_features.json", selected_policy)
    return selected_policy


def run(args):
    path = RESULTS / "run_manifest.json"
    if args.dry_run:
        print("08: Stage06/07 B-MTGNN Structural, L1, 10 paired seeds")
        print("Train 36 origins, dropout-off Train in-sample attribution")
        print("All K=1..157 ranked prefixes; 156 x 10 new fitted seed models")
        print("Unconditional random K=79: 10 subsets x 10 matched fit seeds")
        print("No independent Test, no arbitrary-subset optimality claim")
        return None
    if args.status:
        print(read_json(path) if path.exists() else "08_READY_NO_RUN")
        return None
    if not args.run and not args.ranking_only:
        raise ValueError("Use --run or --ranking-only")
    ctx = upstream()
    spec = contract(ctx)
    sig = signature(spec)
    old = read_json(path) if path.exists() else None
    if old and old["study_signature"] != sig:
        raise RuntimeError("Stage08 code/upstream changed: refuse old caches")
    if old and old["status"] == "PASS":
        print("08_ALREADY_PASS", old["selected_K"])
        return read_json(RESULTS / "selected_features.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    start = time.perf_counter()
    manifest = dict(
        stage=8, status="RUNNING", study_signature=sig,
        contract=spec, started_utc=(old or {}).get(
            "started_utc", datetime.now(timezone.utc).isoformat()),
        test_opened=False,
        resume="py -u Sources/08_feature_selection.py --run")

    write_json(path, manifest)
    ranking, stability = build_ranking(ctx, RESULTS, sig, device)
    if args.ranking_only:
        manifest.update(
            status="TRAIN_RANKING_COMPLETE", ranked_features=len(ranking),
            stability=stability, elapsed_seconds=time.perf_counter() - start)
        write_json(path, manifest)
        return manifest
    order = validate_ranking(ranking, ctx["panel"].features)
    scores = []
    for k in ALL_K:
        current = score_policy(
            ctx, RESULTS, topk_indices(order, k), f"top{k:03d}",
            "TOPK", sig, device)
        scores.append(current)
        print("08_TOPK", k, "MAE", round(current["axis_mae"], 6),
              "PASS", current["qualified"], flush=True)
    chosen = select_k(scores)
    controls = []
    for random_seed in RANDOM_SUBSET_SEEDS:
        current = score_policy(
            ctx, RESULTS, random_indices(random_seed),
            f"random_{random_seed}", "RANDOM", sig, device)
        controls.append(dict(subset_seed=random_seed,
                             axis_mae=current["axis_mae"],
                             ici_mae=current["ici_mae"],
                             K=current["K"]))
    table = pd.DataFrame([
        {key: row[key] for key in (
            "K", "axis_mae", "ici_mae", "profitability_mae",
            "productivity_mae", "growth_mae",
            "2020_mae", "2021_mae", "2022_mae",
            "axis_mae_seed_sd", "ici_mae_seed_sd",
            "mae_improved", "ici_nonworsening",
            "dev_years_improved", "qualified", "parameters")}
        for row in scores])
    table.to_csv(RESULTS / "topk_performance.csv",
                 index=False, encoding="utf-8-sig")
    pd.DataFrame(controls).to_csv(
        RESULTS / "random_k_controls.csv", index=False, encoding="utf-8-sig")
    pareto_table(scores).to_csv(
        RESULTS / "ranked_k_pareto.csv", index=False, encoding="utf-8-sig")
    compare_stage07(scores, RESULTS)
    plots = generate_plots(
        RESULTS, ranking, table, chosen, controls, stability)
    write_selection(ctx, RESULTS, ranking, scores, chosen,
                    controls, stability, sig, plots)
    manifest.update(
        status="PASS", selected_K=chosen,
        complete_K=FEATURE_COUNT,
        new_compact_fit_count=(FEATURE_COUNT - 1) * len(SEEDS),
        random_control_fit_count=len(RANDOM_SUBSET_SEEDS) * len(SEEDS),
        selected_parameters=scores[chosen-1]["parameters"],
        plots=plots, completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - start)
    write_json(path, manifest)
    print("08_COMPLETE", "K", chosen, "157_OF_157", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--ranking-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
