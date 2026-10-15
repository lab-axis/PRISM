"""Stage09 canonical hyperparameter grid and invariant regression (pytest)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from pipeline_support import require_current_results

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prism09_pytest", ROOT / "Sources/09_hyperparameter_tuning.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def ctx():
    require_current_results(8)
    torch.set_num_threads(4)
    return m.upstream()


@pytest.fixture(scope="module")
def economic_bank(ctx):
    return m.components(ctx)


def test_exact_exhaustive_grid_reference_first_and_distinct_alpha_triples():
    assert m.WIDTHS == (8, 16, 32, 64)
    assert m.LEARNING_RATES == (.0003, .001, .003)
    assert m.EPOCHS == (10, 25, 50, 75, 100, 150, 200)
    assert len(m.ALPHAS) == 27
    assert m.REFERENCE == (16, .001, (.25, .1, .25))
    policies = m.grid()
    assert len(policies) == 2268 == len(set(policies))
    assert len({m.policy_id(*args) for args in policies}) == 2268
    trajectory = m.trajectories()
    assert len(trajectory) == 324 == len(set(trajectory))
    assert trajectory[0] == m.REFERENCE
    assert len(trajectory) * len(m.SEEDS) == 3240
    assert 3240 * len(m.EPOCHS) == 22680
    assert all(set(a).issubset(
        set((0., .1, .2, .25, .3, .5, .75, 1.))) for a in m.ALPHAS)
    with pytest.raises(ValueError):
        m.trajectory_id(24, .001, m.REFERENCE[2])
    with pytest.raises(ValueError):
        m.policy_id(*m.REFERENCE, 80)


def test_immutable_06_07_08_chain_and_full_train(ctx):
    count=len(ctx["selected_indices"])
    assert ctx["selected08"]["selected_K"] == count
    assert ctx["selected08"]["selected_original_indices"] == list(ctx["selected_indices"])
    assert ctx["selected08"]["all_ranked_prefixes_exhausted"] is True
    assert set(ctx["selected_indices"]).issubset(ctx["stage07_lock"]["selected_indices"])
    assert ctx["selected_fold"]["train_x"].shape == (len(ctx["m06"].train_origins()),ctx["lookback"],59,count)
    assert ctx["selected_fold"]["val_x"].shape == (12,ctx["lookback"],59,count)
    assert tuple(ctx["full_fold"]["train_origins"]) == tuple(ctx["m06"].train_origins())
    assert tuple(ctx["full_fold"]["val_origins"]) == m.DEV_ORIGINS
    assert ctx["baseline"]["axis_mae"] == pytest.approx(ctx["selected08"]["selected_dev"]["axis_mae"])
    assert ctx["baseline"]["ici_mae"] == pytest.approx(ctx["selected08"]["selected_dev"]["ici_mae"])
    cfg = m.configuration(ctx)
    assert cfg["internal_train_split"] is False
    assert cfg["training_full_admissible_train_only"] is True
    assert cfg["test_opened"] is False
    assert "Data/03_integrated_dataset/raw_test.csv" not in ctx["panel_hashes"]


def test_all_widths_have_actual_forward_and_backward(ctx):
    data = ctx["selected_fold"]
    tensors = {key: torch.as_tensor(data["train_" + key][:2])
               for key in ("x", "mask", "last", "context", "structure")}
    expected = {8: 26926, 16: 67190, 32: 192262, 64: 620582}
    old = ctx["m04"].STAGE04_ACTIVE_WIDTH
    for width in m.WIDTHS:
        model = m.model_factory(ctx,width,2026,data)
        assert sum(p.numel() for p in model.parameters()) == expected[width]-2*width*(157-len(ctx["selected_indices"]))
        assert model.base.values.in_features == len(ctx["selected_indices"])
        assert model.base.values.out_features == width
        assert model.base.missing.out_features == width
        assert model.context_adapter[0].out_features == width
        assert isinstance(model.base.core, ctx["m05"].FixedTemporalInput)
        out = model(tensors["x"], tensors["mask"], torch.eye(59),
                    tensors["last"], tensors["structure"],
                    tensors["context"])
        assert out.shape == (2, 1, 59, 3)
        out.mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in model.parameters())
    assert ctx["m04"].STAGE04_ACTIVE_WIDTH == old  # Never monkeypatch upstream.


def test_alpha_components_change_only_structural_priors(ctx, economic_bank):
    original = m.alpha_prior(ctx, economic_bank, m.REFERENCE[2])
    np.testing.assert_allclose(original, ctx["prior"], atol=1e-10, rtol=0)
    base = ctx["selected_fold"]
    changed = m.alpha_inputs(ctx, original)
    for key in ("train_structure", "val_structure"):
        np.testing.assert_array_equal(changed[key], base[key])
    different = m.alpha_prior(ctx, economic_bank, (.5, .2, .5))
    assert different.shape == (68, 59, 3)
    assert not np.array_equal(different, original)
    candidate = m.alpha_inputs(ctx, different)
    for key in base:
        if key not in ("train_structure", "val_structure","anchor_biases"):
            assert candidate[key] is base[key]
    assert candidate["train_structure"].shape == (len(base["train_origins"]),59,3)
    assert candidate["val_structure"].shape == (12, 59, 3)
    with pytest.raises(ValueError):
        m.alpha_prior(ctx, economic_bank, (.15, .1, .25))


def test_dev_labels_are_not_used_in_model_inputs(ctx, economic_bank):
    prior = m.alpha_prior(ctx, economic_bank, (.5, .1, .25))
    data = m.alpha_inputs(ctx, prior)
    # Use a real copied panel with poisoned 2020–22 labels, not merely
    # an unused temporary array, and reconstruct the alpha-dependent tensors.
    changed_labels = ctx["panel"].y.copy()
    changed_labels[56:68] += 10000.
    assert not np.array_equal(changed_labels[56:68], ctx["panel"].y[56:68])
    poisoned_ctx = dict(ctx, panel=replace(ctx["panel"], y=changed_labels))
    poisoned_prior = m.alpha_prior(
        poisoned_ctx, economic_bank, (.5, .1, .25))
    poisoned_data = m.alpha_inputs(poisoned_ctx, poisoned_prior)
    np.testing.assert_array_equal(prior, poisoned_prior)
    for key in ("train_x", "train_mask", "train_y", "train_structure",
                "val_x", "val_mask", "val_structure", "val_context"):
        np.testing.assert_array_equal(data[key], poisoned_data[key])
    np.testing.assert_array_equal(data["train_y"], ctx["full_fold"]["train_y"])
    np.testing.assert_array_equal(data["val_x"], ctx["selected_fold"]["val_x"])
    center = ctx["panel"].y[:56].mean((0, 1))
    scale = ctx["panel"].y[:56].std((0, 1))
    scale = np.where(scale > 1e-8, scale, 1.)
    np.testing.assert_array_equal(
        data["val_structure"],
        ((prior[56:68] - center) / scale).astype(np.float32))


def test_pairwise_seed_mean_and_candidate_safeguards(ctx):
    truth = ctx["panel"].y[56:68]
    predictions = np.stack([truth + offset for offset in range(-5, 5)])
    direct, member = ctx["m06"].seed_policy_metrics(predictions, ctx["panel"])
    assert direct["axis_mae"] == pytest.approx(2.5)
    assert len(member) == 10
    assert np.abs(predictions.mean(axis=0) - truth).mean() == pytest.approx(.5)
    baseline = dict(axis_mae=2., ici_mae=1.8,
                    **{f"{year}_mae":2. for year in (2020, 2021, 2022)})
    good = dict(axis_mae=1.9, ici_mae=1.79, **{
        "2020_mae":1.8, "2021_mae":1.9, "2022_mae":2.1})
    assert m.qualification(good, baseline)["qualified"]
    assert not m.qualification(dict(good, ici_mae=1.81), baseline)["qualified"]
    assert not m.qualification(dict(good, axis_mae=2.1), baseline)["qualified"]
    assert not m.qualification(dict(
        good, **{"2020_mae":2.1, "2021_mae":2.1}), baseline)["qualified"]


def test_policy_selection_complete_grid_fallback_and_eligible():
    policies = m.grid()
    rows = [dict(policy=m.policy_id(*item), width=item[0],
                 learning_rate=item[1], alpha=list(item[2]),
                 epoch=item[3], parameters=67190,
                 status="PASS", qualified=False, axis_mae=2.)
            for item in policies]
    baseline = m.policy_id(*m.REFERENCE, 75)
    assert m.policy_selection(rows)["policy"] == baseline
    expected = m.policy_id(32, .003, (.5, .2, .5), 50)
    target = next(row for row in rows if row["policy"] == expected)
    target.update(qualified=True, axis_mae=1.9)
    assert m.policy_selection(rows)["policy"] == expected
    target.update(qualified=False, status="NUMERICAL_FAILURE")
    assert m.policy_selection(rows)["policy"] == baseline
    with pytest.raises(ValueError):
        m.policy_selection(rows[:-1])


def test_numpy_original_ici_identity(ctx):
    y = ctx["panel"].y[56:68]
    prior = ctx["prior"][56:68]
    score = ctx["m06"].scores(prior, y,
                             ctx["m04"].past_scores(
                                 ctx["panel"], np.asarray(m.DEV_ORIGINS)))
    np.testing.assert_allclose(
        score["ici_mae"], np.abs((prior - y).mean(-1)).mean(), atol=1e-12)


def test_snapshot_inference_does_not_change_training_trajectory(ctx):
    # The same seed and exact L1 input. Adding one earlier inference
    # snapshot MUST NOT alter the later trained weights/predictions.
    data = ctx["selected_fold"]
    without = m.fit_trajectory(ctx, data, 16, .001, 2026,
                               torch.device("cpu"), snapshots=(10,))
    with_early = m.fit_trajectory(ctx, data, 16, .001, 2026,
                                  torch.device("cpu"), snapshots=(5, 10))
    assert without["failure"] is None and with_early["failure"] is None
    for key in without["states"]["10"]:
        torch.testing.assert_close(
            without["states"]["10"][key],
            with_early["states"]["10"][key],
            rtol=0, atol=0)
    np.testing.assert_array_equal(
        without["predictions"][0], with_early["predictions"][1])


def test_actual_stage06_reference_75_epoch_prediction_parity(ctx):
    outcome = m.fit_trajectory(
        ctx, ctx["selected_fold"], 16, .001, 2026,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        snapshots=(75,))
    assert outcome["failure"] is None
    assert outcome["parameters"] == ctx["selected08"]["selected_parameters"]
    assert m.assert_reference_seed(ctx, outcome, 2026) < m.PARITY_ATOL


def test_hash_validated_restart_and_detection(ctx, tmp_path, monkeypatch):
    forecasts = np.zeros((len(m.EPOCHS), 12, 59, 3), dtype=np.float64)
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(1)
        return dict(epochs=np.array(m.EPOCHS, np.int32),
                    predictions=forecasts,
                    states={str(e): {"placeholder": torch.tensor(1)}
                            for e in m.EPOCHS},
                    failure=None, parameters=ctx["selected08"]["selected_parameters"], seconds=.01)

    monkeypatch.setattr(m, "fit_trajectory", fake_fit)
    for _ in range(2):
        result = m.load_or_fit(
            ctx, tmp_path, *m.REFERENCE, 2026, ctx["selected_fold"],
            "test-signature", "cpu")
        assert result["epochs"].tolist() == list(m.EPOCHS)
    assert len(calls) == 1
    metadata, forecast, checkpoint = m.job_paths(
        tmp_path, *m.REFERENCE, 2026)
    assert all(path.is_file() for path in
               (metadata, forecast, checkpoint))
    with open(forecast, "ab") as fh:
        fh.write(b"tampering")
    with pytest.raises(RuntimeError, match="cache"):
        m.load_or_fit(ctx, tmp_path, *m.REFERENCE, 2026,
                      ctx["selected_fold"], "test-signature", "cpu")


def test_study_is_precommitted_with_no_training_results():
    # Explicitly separate a code-complete implementation from a completed
    # 3,240-trajectory main experiment. No synthetic PASS artifacts.
    assert len(m.trajectories()) * len(m.SEEDS) == 3240
    manifest = m.RESULTS / "run_manifest.json"
    if manifest.is_file():
        state = m.read_json(manifest)
        assert state["status"] in {
            "RUNNING", "READY_TO_RESUME", "PASS"}
        assert state["test_opened"] is False
