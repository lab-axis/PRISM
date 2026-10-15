#!/usr/bin/env python3
"""PRISM Stage09: full-Train Structural B-MTGNN hyperparameter experiment.

Conditional on actual Stage04–08 mode, lookback, preprocessing and retained
inputs, all admissible Train origins and ten paired seeds.
No Train split/CV, no Test. All candidate policies use the same
2020–22 Dev observations. These scores are development-selection results,
NOT independent estimates of generalization or global hyperparameter optima.

The 27 Structural-alpha triples change the economic PRIOR; they do not
multiply the neural correction or alter the graph MixHop alpha. A separate
network is trained for each width/learning-rate/prior combination.

One training trajectory to epoch 200 yields seven isolated 30-draw
MC-dropout snapshots (10,25,50,75,100,150,200). The baseline trajectories
are verified against all ten frozen Stage06 seed forecasts before any
alternative configurations may run. Per-seed MAE is calculated BEFORE the
ten-seed mean, never from an averaged forecast. Incomplete/failed ten-seed
policies are ineligible.

Commands:
    py Sources/09_hyperparameter_tuning.py --dry-run
    py Sources/09_hyperparameter_tuning.py --status
    py -u Sources/09_hyperparameter_tuning.py --run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
from itertools import product
import json
import math
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/09_hyperparameter_tuning.py"
SOURCE08 = ROOT / "Sources/08_feature_selection.py"
RESULT08 = ROOT / "Results/08_feature_selection"
RESULTS = ROOT / "Results/09_hyperparameter_tuning"

WIDTHS = (8, 16, 32, 64)
LEARNING_RATES = (.0003, .001, .003)
EPOCHS = (10, 25, 50, 75, 100, 150, 200)
ALPHAS = tuple(product((0., .25, .5), (0., .1, .2), (0., .25, .5)))
REFERENCE = (16, .001, (.25, .1, .25))
SEEDS = tuple(range(2026, 2036))
DEV_ORIGINS = tuple(range(56, 68))
MODEL = "B-MTGNN"
SCORE_FIELDS = ("axis_mae", "ici_mae", "profitability_mae",
                "productivity_mae", "growth_mae", "2020_mae", "2021_mae",
                "2022_mae", "2020_ici_mae", "2021_ici_mae", "2022_ici_mae")
EPS = 1e-9
PARITY_ATOL = 3e-5

assert len(ALPHAS) == 27 and len(set(ALPHAS)) == 27
assert 4 * 3 * 7 * len(ALPHAS) == 2268
assert REFERENCE[2] in ALPHAS and len(SEEDS) == 10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def stage08():
    spec = importlib.util.spec_from_file_location("_prism09_stage08", SOURCE08)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def upstream() -> dict:
    m08 = stage08()
    ctx = m08.upstream()
    lock = read_json(RESULT08 / "selected_features.json")
    manifest = read_json(RESULT08 / "run_manifest.json")
    indices=tuple(lock.get("selected_original_indices",()))
    if (manifest.get("status") != "PASS"
            or lock.get("status") != "LOCKED"
            or not indices or indices!=tuple(sorted(set(indices)))
            or not set(indices).issubset(ctx["active_indices"])
            or manifest.get("selected_K")!=len(indices)
            or lock.get("selected_K")!=len(indices)
            or lock.get("stage08_signature") != manifest.get("study_signature")
            or manifest.get("contract") != m08.contract(ctx)
            or manifest.get("study_signature") != m08.signature(manifest["contract"])
            or manifest["contract"]["source_sha256"] != sha(SOURCE08)
            or lock.get("stage07_lock_sha256") != ctx["stage07_lock_hash"]
            or not lock.get("all_ranked_prefixes_exhausted")
            or lock.get("test_opened") is not False
            or manifest.get("test_opened") is not False
            or lock.get("lookback")!=ctx["lookback"]
            or lock.get("output_mode")!=ctx["output_mode"]
            or not np.array_equal(ctx["full_fold"]["val_origins"], DEV_ORIGINS)
            or ctx["epochs"] != 75
            or ctx["models"] != (MODEL,)):
        raise RuntimeError("Stage09 upstream Stage06–08 canonical lock mismatch")
    subset=m08.subset(ctx,indices,"stage08_selected")
    data=ctx["m07"].adapt_inputs(ctx["full_fold"],subset,"compact")
    prediction_path=RESULT08/"selected_seed_predictions.npz"
    if lock["selected_predictions_sha256"]!=sha(prediction_path):
        raise RuntimeError("Stage08 selected forecast bank changed")
    with np.load(prediction_path,allow_pickle=False) as saved:
        bank=saved["seed_predictions"].copy()
    baseline,_=ctx["m06"].seed_policy_metrics(bank,ctx["panel"])
    for key in SCORE_FIELDS:
        if abs(float(lock["selected_dev"][key]) -
               float(baseline[key])) > 1e-10:
            raise RuntimeError("Stage08 selected forecast bank differs from its metrics")
    ctx.update(m08=m08, selected08=lock, manifest08=manifest,
               selected_indices=indices,selected_subset=subset,selected_fold=data,
               baseline=baseline,reference_forecasts=bank,
               stage08_lock_hash=sha(RESULT08 / "selected_features.json"),
               stage08_manifest_hash=sha(RESULT08 / "run_manifest.json"))
    return ctx


def configuration(ctx: dict) -> dict:
    m04 = ctx["m04"]
    return dict(
        stage="PRISM09_FULL_TRAIN_BMTGNN_HYPERPARAMETER_GRID",
        source_sha256=sha(SOURCE),
        stage08_source_sha256=sha(SOURCE08),
        stage08_lock_sha256=ctx["stage08_lock_hash"],
        stage08_manifest_sha256=ctx["stage08_manifest_hash"],
        stage07_lock_sha256=ctx["stage07_lock_hash"],
        stage06_lock_sha256=ctx["upstream06_lock_sha256"],
        stage06_manifest_sha256=ctx["upstream06_manifest_sha256"],
        stage06_input_sha256=ctx["upstream06_input_sha256"],
        source_component_sha256=sha(m04.STAGE04_COMPONENT_SOURCE),
        raw_panel_hashes=ctx["panel_hashes"],
        model=MODEL,output_mode=ctx["output_mode"],lookback=ctx["lookback"],
        selected_feature_indices=list(ctx["selected_indices"]),
        selected_input_sha256=ctx["m07"].dataset_sha(ctx["selected_fold"]),
        hidden_widths=list(WIDTHS), learning_rates=list(LEARNING_RATES),
        snapshots=list(EPOCHS),
        alpha_triples=[list(alpha) for alpha in active_alphas(ctx["output_mode"])],
        structural_alpha_active=ctx["output_mode"]=="structural",
        policy_count=len(grid(ctx["output_mode"])),
        baseline_width=16, baseline_lr=.001, baseline_epoch=75,
        baseline_alpha=list(REFERENCE[2]),
        training_origins=ctx["selected_fold"]["train_origins"].tolist(),
        dev_origins=list(DEV_ORIGINS), seeds=list(SEEDS),
        training_full_admissible_train_only=True,
        internal_train_split=False, test_opened=False,
        preprocessing=ctx["locked_policy"].id,
        fixed_core_slots=20, fixed_dropout=.1,
        fixed_graph_top_k=8, fixed_graph_embedding=8,
        fixed_weight_decay=.0001, fixed_grad_clip=5.,
        snapshot_inference_draws=30,
        score_protocol="TEN_INDIVIDUAL_SEED_MAE_THEN_MEAN",
        selection="MAE_IMPROVE_ICI_NONWORSE_TWO_OF_THREE_YEARS",
        dev_reused_stages_04_through_08=True,
        best_in_grid_not_global=True, no_legacy_cache=True)


def signature(configuration: dict) -> str:
    return hashlib.sha256(json.dumps(
        configuration, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def trajectory_id(width: int, lr: float, alpha: tuple[float, ...]) -> str:
    if width not in WIDTHS or lr not in LEARNING_RATES or alpha not in ALPHAS:
        raise ValueError("Unknown Stage09 configuration")
    return (f"w{width:02d}_lr{round(lr * 1000000):04d}_a"
            + "_".join(f"{round(a * 100):03d}" for a in alpha))


def policy_id(width: int, lr: float, alpha: tuple[float, ...], epoch: int) -> str:
    if epoch not in EPOCHS:
        raise ValueError("Unknown Stage09 snapshot")
    return trajectory_id(width, lr, alpha) + f"_e{epoch:03d}"


def active_alphas(mode):
    """Alpha changes only a Structural anchor, not Direct/Residual models."""
    if mode not in ("direct","residual","structural"):
        raise ValueError("Unknown Stage04-selected formulation")
    return ALPHAS if mode=="structural" else (REFERENCE[2],)


def grid(mode="structural"):
    return tuple((width, lr, alpha, epoch)
                 for width, lr, alpha, epoch in
                 product(WIDTHS, LEARNING_RATES, active_alphas(mode), EPOCHS))


def trajectories(mode="structural"):
    """Reference first; no alternative evaluated before reference parity."""
    reference = REFERENCE
    rest = tuple(item for item in product(WIDTHS, LEARNING_RATES, active_alphas(mode))
                 if item != reference)
    return (reference,) + rest


def model_factory(ctx, width: int, seed: int, data: dict):
    """Stage05/06 exact reference at W16; pure variable-width replacements.

    Do not monkeypatch mutable Stage04 module constants.
    """
    if width not in WIDTHS or seed not in SEEDS:
        raise ValueError("Unregistered model/seed")
    count=len(ctx["selected_indices"])
    net=ctx["m07"].instantiate_subset(
        ctx,ctx["selected_subset"],seed,compact=count<157)
    if width != 16:
        m04, m05 = ctx["m04"], ctx["m05"]
        net.base = m04.ForecastModel(
            MODEL,count,59,width,20,1)
        net.base.core = m05.FixedTemporalInput(net.base.core)
        net.context_adapter = nn.Sequential(
            nn.Linear(9, width), nn.SiLU(), nn.Linear(width, width))
    # Re-estimate the selected anchor's median for each candidate alpha,
    # and zero the output AFTER replacing the width-dependent core.
    ctx["m04"].initialize_calibrated_model(net,data)
    expected=({8:26926,16:67190,32:192262,64:620582}[width]
              -2*width*(157-count))
    if sum(parameter.numel() for parameter in net.parameters()) != expected:
        raise RuntimeError("Stage09 expected model parameter count changed")
    return net


def components(ctx: dict):
    return ctx["m04"].economic_components(ctx["panel"])


def alpha_prior(ctx: dict, component_bank, alpha: tuple[float, ...]):
    if alpha not in ALPHAS:
        raise ValueError("Unregistered Structural-alpha triple")
    n, d, info = component_bank
    prior = ctx["m04"].component_structural_prior(
        n, d, info, np.arange(68), np.asarray(alpha, dtype=np.float64))
    if prior.shape != (68, 59, 3) or not np.isfinite(prior).all():
        raise RuntimeError("Invalid structural prior")
    if alpha == REFERENCE[2] and not np.allclose(
            prior, ctx["prior"], atol=1e-10, rtol=0):
        raise RuntimeError("Stage09 original alpha does not reproduce Stage04")
    return prior


def alpha_inputs(ctx: dict, prior: np.ndarray):
    """Only Structural prior differs; original target labels stay fixed."""
    base = ctx["selected_fold"]
    result = dict(base)
    # Stage06 normalized with float64 TRAIN statistics before exporting
    # float32 buffers. Never re-normalize using those rounded buffers.
    if np.allclose(prior, ctx["prior"], atol=1e-10, rtol=0):
        result["train_structure"] = base["train_structure"]
        result["val_structure"] = base["val_structure"]
    else:
        center = ctx["panel"].y[:56].mean((0, 1))
        scale = ctx["panel"].y[:56].std((0, 1))
        scale = np.where(scale > 1e-8, scale, 1.)
        result["train_structure"] = (
            (prior[base["train_origins"]] - center) / scale).astype(np.float32)
        result["val_structure"] = (
            (prior[list(DEV_ORIGINS)] - center) / scale).astype(np.float32)
        result["anchor_biases"]=ctx["m04"].train_anchor_biases(
            ctx["panel"],prior,base["train_origins"],center,scale)
    if (result["train_structure"].shape != (len(base["train_origins"]),59,3)
            or result["val_structure"].shape != (12, 59, 3)
            or not all(np.isfinite(result[key]).all()
                       for key in ("train_structure", "val_structure"))):
        raise RuntimeError("Invalid alpha-dependent structural tensor")
    for key, value in base.items():
        if key not in ("train_structure", "val_structure","anchor_biases") and (
                result[key] is not value):
            raise RuntimeError("Stage09 changed a non-Structural input")
    if np.allclose(prior, ctx["prior"], atol=1e-10, rtol=0):
        for key in ("train_structure", "val_structure"):
            if not np.array_equal(result[key], base[key]):
                raise RuntimeError("Reference Structural tensor parity failed")
    return result


def fit_trajectory(ctx: dict, data: dict, width: int, lr: float, seed: int,
                   device, snapshots=EPOCHS):
    """One seeded trajectory, with no optimizer/RNG mutation from snapshots."""
    snapshots = tuple(snapshots)
    if (width not in WIDTHS or lr not in LEARNING_RATES or seed not in SEEDS
            or not snapshots or snapshots != tuple(sorted(set(snapshots)))
            or snapshots[0] < 1 or snapshots[-1] > 200):
        raise ValueError("Invalid Stage09 training trajectory")
    m04 = ctx["m04"]
    model = model_factory(ctx,width,seed,data).to(device)
    params = sum(p.numel() for p in model.parameters())
    keys = ("train_x", "train_mask", "train_y", "train_last",
            "train_context", "train_structure", "val_x", "val_mask",
            "val_last", "val_context", "val_structure")
    z = {key: torch.as_tensor(data[key], device=device) for key in keys}
    graph = torch.eye(59, device=device)
    yscale = torch.as_tensor(data["y_scale"], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=.0001)
    forecasts, states, completed = [], {}, []
    failure = None
    started = time.perf_counter()
    for epoch in range(1, snapshots[-1] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(z["train_x"], z["train_mask"], graph,
                       z["train_last"], z["train_structure"],
                       z["train_context"])
        loss = m04.joint_axis_ici_loss(output,z["train_y"],yscale)
        if not torch.isfinite(loss):
            failure = f"Nonfinite Train loss at epoch {epoch}"
            break
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5., error_if_nonfinite=True)
        except RuntimeError as exc:
            if not any(word in str(exc).lower() for word in
                       ("nonfinite", "non-finite", "nan", "infinite")):
                raise
            failure = f"Nonfinite gradient at epoch {epoch}: {exc}"
            break
        optimizer.step()
        if epoch not in snapshots:
            continue
        with m04.isolated_snapshot_inference(model):
            prediction = m04.mtgnn_bayesian_inference(
                model, z, graph, MODEL, seed,
                data["y_center"], data["y_scale"])["prediction"]
        if not np.isfinite(prediction).all():
            failure = f"Nonfinite Dev prediction at epoch {epoch}"
            break
        states[str(epoch)] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}
        completed.append(epoch)
        forecasts.append(prediction)
    elapsed = time.perf_counter() - started
    del optimizer, model, z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = (np.stack(forecasts) if forecasts else
             np.empty((0, 12, 59, 3), dtype=np.float64))
    return dict(
        epochs=np.asarray(completed, dtype=np.int32),
        predictions=saved, states=states,
        parameters=params, seconds=float(elapsed), failure=failure)


def job_paths(root: Path, width: int, lr: float,
              alpha: tuple[float, ...], seed: int):
    folder = root / "trajectories" / trajectory_id(width, lr, alpha)
    stem = f"B-MTGNN__s{seed}"
    return tuple(folder / (stem + suffix)
                 for suffix in (".json", ".npz", ".pt"))


def load_or_fit(ctx: dict, root: Path, width: int, lr: float,
                alpha: tuple[float, ...], seed: int, data: dict,
                study_sig: str, device):
    metadata_file, pred_file, weight_file = job_paths(
        root, width, lr, alpha, seed)
    label = trajectory_id(width, lr, alpha)
    if all(path.is_file() for path in
           (metadata_file, pred_file, weight_file)):
        record = read_json(metadata_file)
        if (record.get("signature") != study_sig
                or record.get("trajectory") != label
                or record.get("seed") != seed
                or record.get("lookback") != ctx["lookback"]
                or record.get("output_mode") != ctx["output_mode"]
                or record.get("selected_indices") != list(ctx["selected_indices"])
                or record.get("epochs_requested") != list(EPOCHS)
                or record.get("pred_sha256") != sha(pred_file)
                or record.get("checkpoint_sha256") != sha(weight_file)):
            raise RuntimeError("Stale/invalid Stage09 trajectory cache")
        with np.load(pred_file, allow_pickle=False) as saved:
            epochs = saved["epochs"].copy()
            predictions = saved["predictions"].copy()
        if (predictions.shape != (len(epochs), 12, 59, 3)
                or not np.isfinite(predictions).all()
                or not set(epochs).issubset(EPOCHS)
                or epochs.tolist() != sorted(set(epochs.tolist()))
                or record.get("epochs_complete") != epochs.tolist()):
            raise RuntimeError("Malformed Stage09 saved trajectory predictions")
        return dict(epochs=epochs, predictions=predictions,
                    failure=record["failure"],
                    parameters=record["parameters"])
    if any(path.exists() for path in
           (metadata_file, pred_file, weight_file)):
        raise RuntimeError("Partial Stage09 trajectory files; refuse mixed cache")
    outcome = fit_trajectory(
        ctx, data, width, lr, seed, device, snapshots=EPOCHS)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(
        signature=study_sig, trajectory=label, seed=seed,
        width=width, learning_rate=lr, alpha=list(alpha),
        lookback=ctx["lookback"],output_mode=ctx["output_mode"],
        selected_indices=list(ctx["selected_indices"]),
        epoch_snapshots=outcome["epochs"].tolist(),
        parameters=outcome["parameters"], states=outcome["states"])
    writing = weight_file.with_suffix(".pt.writing")
    torch.save(checkpoint, writing)
    os.replace(writing, weight_file)
    atomic_npz(pred_file, epochs=outcome["epochs"],
               predictions=outcome["predictions"])
    atomic_json(metadata_file, dict(
        signature=study_sig, trajectory=label, seed=seed,
        width=width, learning_rate=lr, alpha=list(alpha),
        model=MODEL,lookback=ctx["lookback"],output_mode=ctx["output_mode"],
        selected_indices=list(ctx["selected_indices"]),
        epochs_requested=list(EPOCHS),
        epochs_complete=outcome["epochs"].tolist(),
        parameters=outcome["parameters"],
        pred_sha256=sha(pred_file),
        checkpoint_sha256=sha(weight_file),
        seconds=outcome["seconds"], failure=outcome["failure"],
        mc_dropout_draws=30, test_opened=False))
    return outcome


def assert_reference_seed(ctx: dict, result: dict, seed: int):
    epochs = result["epochs"].tolist()
    if 75 not in epochs or result["parameters"] != ctx["selected08"]["selected_parameters"]:
        raise RuntimeError(f"Stage09 baseline seed {seed} lacks epoch 75")
    index = epochs.index(75)
    actual = result["predictions"][index]
    expected = ctx["reference_forecasts"][SEEDS.index(seed)]
    if (actual.shape != (12, 59, 3)
            or not np.allclose(actual, expected, atol=PARITY_ATOL, rtol=0)):
        maximum = float(np.max(np.abs(actual - expected)))
        raise RuntimeError(
            f"Stage09 BASELINE PARITY FAIL seed {seed}; max difference {maximum}")
    return float(np.max(np.abs(actual - expected)))


def baseline_parity(ctx: dict, root: Path, study_sig: str):
    folder = root / "trajectories" / trajectory_id(*REFERENCE)
    forecasts = []
    differences = []
    for seed in SEEDS:
        meta = read_json(folder / f"B-MTGNN__s{seed}.json")
        path = folder / f"B-MTGNN__s{seed}.npz"
        checkpoint = folder / f"B-MTGNN__s{seed}.pt"
        if (meta["signature"] != study_sig
                or meta["pred_sha256"] != sha(path)
                or meta["checkpoint_sha256"] != sha(checkpoint)):
            raise RuntimeError("Stage09 baseline checkpoint provenance mismatch")
        with np.load(path, allow_pickle=False) as saved:
            result = dict(epochs=saved["epochs"].copy(),
                          predictions=saved["predictions"].copy(),
                          parameters=meta["parameters"])
        differences.append(assert_reference_seed(ctx, result, seed))
        forecasts.append(
            result["predictions"][result["epochs"].tolist().index(75)])
    scores, _ = ctx["m06"].seed_policy_metrics(
        np.stack(forecasts), ctx["panel"])
    if any(abs(scores[key] - ctx["baseline"][key]) > PARITY_ATOL
           for key in SCORE_FIELDS):
        raise RuntimeError("Stage09 baseline 10-seed metrics differ from selected Stage08")
    return dict(max_absolute_prediction_diff=max(differences),
                max_absolute_metric_diff=max(abs(
                    scores[key] - ctx["baseline"][key]) for key in SCORE_FIELDS),
                all_ten_seed_forecasts_checked=True,
                reference_parameters=ctx["selected08"]["selected_parameters"],
                tolerance=PARITY_ATOL)


def qualification(row: dict, baseline: dict):
    annual_wins = sum(
        row[f"{year}_mae"] < baseline[f"{year}_mae"] - EPS
        for year in (2020, 2021, 2022))
    improvement = row["axis_mae"] < baseline["axis_mae"] - EPS
    ici = row["ici_mae"] <= baseline["ici_mae"] + EPS
    return dict(
        three_axis_improved=bool(improvement),
        ici_nonworsening=bool(ici),
        years_improved=int(annual_wins),
        qualified=bool(improvement and ici and annual_wins >= 2))


def policy_selection(rows: list[dict],mode="structural") -> dict:
    if (len(rows) != len(grid(mode)) or
            len({r["policy"] for r in rows}) != len(rows)):
        raise ValueError("Need every unique policy in the active grid")
    eligible = [r for r in rows
                if r["status"] == "PASS" and r["qualified"]]
    baseline_id = policy_id(*REFERENCE, 75)
    reference = next((r for r in rows if r["policy"] == baseline_id), None)
    if reference is None or reference["status"] != "PASS":
        raise RuntimeError("Incomplete Stage09 baseline policy")
    if not eligible:
        return reference

    def preference(r):
        distance = sum((a - b)**2 for a, b in
                       zip(r["alpha"], REFERENCE[2]))
        return (r["axis_mae"], r["parameters"], r["epoch"],
                abs(math.log(r["learning_rate"] / REFERENCE[1])),
                distance, r["policy"])

    return min(eligible, key=preference)


def scoring_table(ctx: dict, root: Path, study_sig: str) -> list[dict]:
    """Reproduce ALL scored policies from persisted individual forecasts."""
    baseline = ctx["baseline"]
    records = []
    for width, lr, alpha in product(WIDTHS, LEARNING_RATES,active_alphas(ctx["output_mode"])):
        entries = []
        for seed in SEEDS:
            meta, path, weight = job_paths(root, width, lr, alpha, seed)
            record = read_json(meta)
            if (record.get("signature") != study_sig
                    or record.get("pred_sha256") != sha(path)
                    or record.get("checkpoint_sha256") != sha(weight)
                    or record.get("seed") != seed):
                raise RuntimeError("Missing/corrupted Stage09 ten-seed score input")
            with np.load(path, allow_pickle=False) as saved:
                entries.append(dict(
                    epochs=saved["epochs"].tolist(),
                    predictions=saved["predictions"].copy(),
                    failure=record["failure"],
                    parameters=record["parameters"]))
        for epoch in EPOCHS:
            policy = policy_id(width, lr, alpha, epoch)
            common = dict(
                policy=policy, width=width, learning_rate=lr,
                alpha=list(alpha), alpha_profitability=alpha[0],
                alpha_productivity=alpha[1], alpha_growth=alpha[2],
                epoch=epoch, parameters=entries[0]["parameters"],
                completed_seed_count=sum(epoch in x["epochs"] for x in entries))
            if common["completed_seed_count"] != len(SEEDS):
                records.append(dict(**common, status="NUMERICAL_FAILURE",
                                    qualified=False))
                continue
            forecast = np.stack([
                entry["predictions"][entry["epochs"].index(epoch)]
                for entry in entries])
            metrics, _ = ctx["m06"].seed_policy_metrics(
                forecast, ctx["panel"])
            gate = qualification(metrics, baseline)
            records.append(dict(**common, status="PASS",
                                **metrics, **gate))
    return records


def economic_only(ctx: dict, component_bank):
    truth = ctx["panel"].y[list(DEV_ORIGINS)]
    past = ctx["m04"].past_scores(ctx["panel"], np.asarray(DEV_ORIGINS))
    records = []
    origins=ctx["selected_fold"]["train_origins"]
    for alpha in active_alphas(ctx["output_mode"]):
        prior = alpha_prior(ctx, component_bank, alpha)
        metrics = ctx["m06"].scores(prior[list(DEV_ORIGINS)], truth, past)
        bias=np.median(ctx["panel"].y[origins]-prior[origins],axis=0)
        calibrated=ctx["m06"].scores(prior[list(DEV_ORIGINS)]+bias,truth,past)
        records.append(dict(alpha=list(alpha),
                            alpha_profitability=alpha[0],
                            alpha_productivity=alpha[1],
                            alpha_growth=alpha[2], **metrics,
                            **{"calibrated_"+key:value for key,value in calibrated.items()}))
    return records


def plots(root: Path, records: list[dict], selected: dict,
          economic: list[dict]):
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    scored = pd.DataFrame([r for r in records if r["status"] == "PASS"])
    figures = []

    def save(fig, title):
        fig.tight_layout()
        file = output / (title + ".png")
        fig.savefig(file, dpi=170, bbox_inches="tight")
        plt.close(fig)
        figures.append("plots/" + file.name)

    fig, ax = plt.subplots(figsize=(10, 5))
    for width in WIDTHS:
        data = scored[scored.width == width].groupby("epoch").axis_mae.mean()
        ax.plot(data.index, data.values, marker="o", label=f"W={width}")
    ax.set(xlabel="Epoch snapshot", ylabel="Average Dev three-axis MAE",
           title="Width–epoch descriptive mean over LR and alpha")
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, "09_01_width_epoch")

    fig, ax = plt.subplots(figsize=(9, 5))
    matrix = scored.pivot_table(
        index="width", columns="learning_rate", values="axis_mae", aggfunc="mean")
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str))
    ax.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    ax.set(xlabel="Learning rate", ylabel="Width",
           title="Mean Dev MAE over alpha and epoch (not a model selection)")
    fig.colorbar(im, ax=ax, label="MAE")
    save(fig, "09_02_width_learning_rate")

    fig, ax = plt.subplots(figsize=(10, 5))
    by_alpha = scored.groupby(
        ["alpha_profitability", "alpha_productivity",
         "alpha_growth"], as_index=False).axis_mae.mean()
    scatter = ax.scatter(
        by_alpha.alpha_profitability, by_alpha.alpha_growth,
        c=by_alpha.axis_mae, s=80 + 200 * by_alpha.alpha_productivity,
        cmap="viridis", edgecolors="black", linewidth=.4)
    ax.set(xlabel="Structural alpha: profitability",
           ylabel="Structural alpha: growth",
           title="Alpha triple sensitivity; marker area encodes productivity alpha")
    fig.colorbar(scatter, ax=ax, label="Mean Dev MAE")
    save(fig, "09_03_structural_alpha")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = scored.width.to_numpy()
    ax.scatter(scored.axis_mae, scored.ici_mae,
               c=colors, cmap="viridis", s=9, alpha=.5)
    ax.scatter([selected["axis_mae"]], [selected["ici_mae"]],
               marker="X", s=100, color="black", label="Selected")
    ax.set(xlabel="Three-axis MAE", ylabel="ICI MAE",
           title="All numerically feasible bounded-grid policies")
    ax.legend()
    save(fig, "09_04_mae_ici_tradeoff")

    reference = next(r for r in records if
                     r["policy"] == policy_id(*REFERENCE, 75))
    comparison = [reference["axis_mae"], selected["axis_mae"]]
    labels = ["Original B-MTGNN", "Selected B-MTGNN"]
    econ = next(r for r in economic if r["alpha"] == selected["alpha"])
    comparison.append(econ["axis_mae"])
    labels.append("Economic-only (selected alpha)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, comparison)
    ax.set(ylabel="Dev 3-axis MAE",
           title="Different priors compared only with matching alpha labels")
    save(fig, "09_05_selected_comparison")
    return figures


def write_final(ctx: dict, root: Path, records: list[dict],
                chosen: dict, economic: list[dict], study_sig: str,
                parity: dict):
    frame = pd.DataFrame(records)
    frame.to_csv(root / f"dev_all_{len(records)}.csv",
                 index=False, encoding="utf-8-sig")
    econ = pd.DataFrame(economic)
    econ.to_csv(root / "economic_only_alpha.csv",
                index=False, encoding="utf-8-sig")
    selection = dict(
        stage=9, status="LOCKED", study_signature=study_sig,
        selected_policy=chosen["policy"],
        selected_width=chosen["width"],
        selected_learning_rate=chosen["learning_rate"],
        selected_epoch=chosen["epoch"],
        selected_structural_alpha=chosen["alpha"],
        selected_parameters=chosen["parameters"],
        eligible_policy_count=sum(bool(r["qualified"]) for r in records),
        numeric_infeasible_policy_count=sum(
            r["status"] != "PASS" for r in records),
        evaluated_grid_count=len(records), all_policies_present=True,
        scope="BEST_ELIGIBLE_OF_ACTIVE_GRID_NOT_GLOBAL",
        output_mode=ctx["output_mode"],lookback=ctx["lookback"],
        selected_original_indices=list(ctx["selected_indices"]),
        learning_recipe=ctx["m04"].LEARNING_RECIPE,
        baseline_parity=parity,
        baseline_dev={key: ctx["baseline"][key] for key in SCORE_FIELDS},
        selected_dev={key: chosen[key] for key in SCORE_FIELDS},
        selected_vs_economic_only_matching_alpha=next(
            r for r in economic if r["alpha"] == chosen["alpha"]),
        selection_informed_by_reused_dev=True,
        independent_generalization_not_established=True,
        test_opened=False,
        selected_seed_checkpoints=[
            str(job_paths(root, chosen["width"],
                          chosen["learning_rate"], tuple(chosen["alpha"]), seed)[2]
                .relative_to(root)).replace("\\", "/")
            for seed in SEEDS],
        selected_snapshot_epoch=chosen["epoch"],
        completed_utc=datetime.now(timezone.utc).isoformat())
    figures = plots(root, records, chosen, economic)
    selection["figures"] = figures
    atomic_json(root / "dev_selection.json", selection)
    rows = []
    reference = next(r for r in records if
                     r["policy"] == policy_id(*REFERENCE, 75))
    for label, row in (("original_BMTGNN", reference),
                       ("selected_BMTGNN", chosen)):
        rows.append(dict(comparator=label, width=row["width"],
                         learning_rate=row["learning_rate"],
                         epoch=row["epoch"], alpha=row["alpha"],
                         **{key: row[key] for key in SCORE_FIELDS}))
    for label, alpha in (("economic_original_alpha", REFERENCE[2]),
                         ("economic_selected_alpha", tuple(chosen["alpha"]))):
        econ = next(r for r in economic if r["alpha"] == list(alpha))
        rows.append(dict(comparator=label, width=None,
                         learning_rate=None, epoch=None,
                         alpha=list(alpha),
                         **{key: econ[key] for key in SCORE_FIELDS}))
        rows.append(dict(comparator=label+"_train_median",width=None,
                         learning_rate=None,epoch=None,alpha=list(alpha),
                         **{key:econ["calibrated_"+key] for key in SCORE_FIELDS}))
    pd.DataFrame(rows).to_csv(
        root / "dev_comparison.csv", index=False, encoding="utf-8-sig")
    return selection


def run(args):
    if args.restart and not args.run:
        raise ValueError("--restart requires --run")
    if args.dry_run:
        print("09: bounded Structural B-MTGNN hyperparameter study")
        print("Width", WIDTHS, "LR", LEARNING_RATES, "Epochs", EPOCHS)
        print("Alpha triples", len(ALPHAS),
              "Policies", len(grid()), "Fitting trajectories",
              len(trajectories()) * len(SEEDS))
        print("Train: Stage08's actual full-Train inputs; Dev: 12 quarters; NO CV/TEST")
        print("Structural mode: 2268 policies; other modes: alpha inactive, 84 policies")
        return None
    manifest_file = RESULTS / "run_manifest.json"
    if args.status:
        print(read_json(manifest_file) if manifest_file.exists()
              else "09_READY_NO_RUN")
        return None
    if not args.run:
        raise ValueError("Use --run")
    ctx = upstream()
    spec = configuration(ctx)
    study_sig = signature(spec)
    if args.restart:
        ctx["m04"].reset_stage_results(RESULTS)
    previous = read_json(manifest_file) if manifest_file.exists() else None
    if previous and previous["signature"] != study_sig:
        raise RuntimeError("Stage09 upstream/source changed; refuse legacy cache")
    if previous and previous["status"] == "PASS":
        print("09_ALREADY_COMPLETE", previous["selected_policy"])
        return previous
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    config_file = RESULTS / "design.json"
    if config_file.exists():
        if read_json(config_file) != spec:
            raise RuntimeError("Stage09 frozen experiment design changed")
    else:
        atomic_json(config_file, spec)
    manifest = dict(
        stage=9, status="RUNNING", signature=study_sig,
        design_sha256=sha(config_file),
        started_utc=(previous or {}).get(
            "started_utc", datetime.now(timezone.utc).isoformat()),
        target_trajectories=len(trajectories(ctx["output_mode"])) * len(SEEDS),
        completed_trajectories=(previous or {}).get(
            "completed_trajectories", 0),
        test_opened=False,
        resume="py -u Sources/09_hyperparameter_tuning.py --run")
    atomic_json(manifest_file, manifest)
    if previous:
        print("09_RESUME_SAVED",
              previous.get("completed_trajectories", 0), "/",
              manifest["target_trajectories"], flush=True)

    bank = components(ctx)
    prepared = {}
    completed = 0
    parity = None
    for index, (width, lr, alpha) in enumerate(trajectories(ctx["output_mode"])):
        if alpha not in prepared:
            prepared[alpha] = alpha_inputs(ctx, alpha_prior(ctx, bank, alpha))
        for seed in SEEDS:
            cached = job_paths(
                RESULTS, width, lr, alpha, seed)[0].is_file()
            outcome = load_or_fit(
                ctx, RESULTS, width, lr, alpha, seed,
                prepared[alpha], study_sig, device)
            if index == 0:
                assert_reference_seed(ctx, outcome, seed)
            completed += 1
            if not cached or completed % 100 == 0:
                print("09_TRAJECTORY", completed, "/",
                      manifest["target_trajectories"],
                      "CACHED" if cached else "TRAINED",
                      trajectory_id(width, lr, alpha), seed,
                      "EPOCHS", outcome["epochs"].tolist(),
                      "FAILURE", outcome["failure"], flush=True)
                manifest["completed_trajectories"] = completed
                atomic_json(manifest_file, manifest)
        if index == 0:
            parity = baseline_parity(ctx, RESULTS, study_sig)
            manifest["reference_parity"] = parity
            atomic_json(manifest_file, manifest)
            print("09_REFERENCE_PARITY_PASS", parity, flush=True)
    records = scoring_table(ctx, RESULTS, study_sig)
    chosen = policy_selection(records,ctx["output_mode"])
    economic = economic_only(ctx, bank)
    selection = write_final(
        ctx, RESULTS, records, chosen, economic, study_sig, parity)
    manifest.update(
        status="PASS",
        completed_trajectories=completed,
        evaluated_policy_count=len(records),
        selected_policy=chosen["policy"],
        selected_width=chosen["width"],
        selected_epoch=chosen["epoch"],
        selected_structural_alpha=chosen["alpha"],
        eligible_policy_count=selection["eligible_policy_count"],
        numeric_infeasible_policy_count=
        selection["numeric_infeasible_policy_count"],
        selection_sha256=sha(RESULTS / "dev_selection.json"),
        elapsed_seconds=time.perf_counter() - start,
        completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(manifest_file, manifest)
    print("09_COMPLETE", chosen["policy"],
          "DEV_MAE", chosen["axis_mae"], "ICI", chosen["ici_mae"],
          flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--restart",action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
