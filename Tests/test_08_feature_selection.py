"""Pytest regression for the single canonical Stage08 B-MTGNN study."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prism08_pytest", ROOT / "Sources/08_feature_selection.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def ctx():
    torch.set_num_threads(4)
    return m.upstream()


def test_pytest_single_canonical_grid_and_random_control():
    assert m.SEEDS == tuple(range(2026, 2036))
    assert m.TRAIN_ORIGINS == tuple(range(20, 56))
    assert m.ALL_K == tuple(range(1, 158))
    assert m.TRAIN_BLOCKS == ((20, 32), (32, 44), (44, 56))
    assert m.RANDOM_K == 79
    assert len(m.RANDOM_SUBSET_SEEDS) == 10
    assert not hasattr(m, "COARSE_K")
    assert not hasattr(m, "expand_fine_grid")


def test_complete_stage07_06_lock(ctx):
    assert ctx["stage07_lock"]["selected_policy"] == "no_removal"
    assert ctx["stage07_lock"]["selected_indices"] == list(range(157))
    assert ctx["models"] == ("B-MTGNN",)
    assert ctx["epochs"] == 75
    assert ctx["full_fold"]["train_x"].shape == (36, 1, 59, 157)
    assert ctx["full_fold"]["val_x"].shape == (12, 1, 59, 157)
    assert ctx["original_forecasts"].shape == (10, 12, 59, 3)
    assert ctx["baseline"]["axis_mae"] == pytest.approx(2.451713824742347)
    assert ctx["baseline"]["ici_mae"] == pytest.approx(1.8255138450145005)
    assert ctx["stage07_lock"]["test_opened"] is False
    assert "Data/03_integrated_dataset/raw_test.csv" not in ctx["panel_hashes"]
    contract = m.contract(ctx)
    assert contract["ranked_prefixes"] == list(range(1, 158))
    assert contract["per_seed_metric_then_mean"] is True
    assert contract["independent_test"] is False


def test_drops_noninput_paths_unchanged(ctx):
    data = ctx["full_fold"]
    for k in (1, 79, 156):
        indices = tuple(sorted(range(k)))
        chosen = m.subset(ctx, indices, f"test{k}")
        compact = ctx["m07"].adapt_inputs(data, chosen, "compact")
        assert compact["train_x"].shape == (36, 1, 59, k)
        assert compact["val_x"].shape == (12, 1, 59, k)
        np.testing.assert_array_equal(
            compact["train_x"], data["train_x"][..., list(indices)])
        np.testing.assert_array_equal(
            compact["val_mask"], data["val_mask"][..., list(indices)])
        for key in ("train_y", "val_y", "train_last", "val_last",
                    "train_context", "val_context",
                    "train_structure", "val_structure",
                    "train_origins", "val_origins", "y_center", "y_scale"):
            np.testing.assert_array_equal(compact[key], data[key])
    with pytest.raises(ValueError):
        m.subset(ctx, (5, 2), "bad")
    with pytest.raises(ValueError):
        m.subset(ctx, (), "bad")


def test_frozen_stage06_checkpoint_is_real_l1_and_eval_only(ctx):
    model = m.load_frozen_seed(ctx, 2026, "cpu")
    assert model.training is False
    assert model.architecture == "B-MTGNN"
    assert model.prediction_mode == "structural"
    assert model.base.values.in_features == 157
    assert isinstance(model.base.core, ctx["m05"].FixedTemporalInput)
    assert sum(p.numel() for p in model.parameters()) == 67190
    z, truth = m.ranking_tensors(ctx, "cpu")
    assert z["x"].shape == (36, 1, 59, 157)
    assert truth.shape == (36, 59, 3)
    np.testing.assert_array_equal(truth, ctx["panel"].y[20:56])
    with torch.inference_mode():
        out = model(
            z["x"][:1], z["mask"][:1], torch.eye(59),
            z["last"][:1], z["structure"][:1], z["context"][:1])
    assert out.shape == (1, 1, 59, 3)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        m.load_frozen_seed(ctx, 9999, "cpu")


def test_ranking_train_inputs_not_poisoned_by_dev(ctx):
    z, labels = m.ranking_tensors(ctx, "cpu")
    before = {key: value.clone() for key, value in z.items()}
    copy = ctx["panel"].y.copy()
    copy[56:] += 1e9
    np.testing.assert_array_equal(
        labels, copy[np.asarray(m.TRAIN_ORIGINS)])
    for key in z:
        torch.testing.assert_close(z[key], before[key])


def test_mask_both_values_and_missing_without_mutating_inputs():
    class Tiny(torch.nn.Module):
        def forward(self, x, mask, graph, last, prior, context):
            score = x[:, 0].sum(-1) + mask[:, 0].sum(-1)
            return score[:, None, :, None].expand(-1, 1, -1, 3)

    net = Tiny().eval()
    x = torch.zeros((36, 1, 59, 157))
    mask = torch.zeros_like(x)
    x[..., 0] = 1
    mask[..., 0] = 1
    z = dict(x=x, mask=mask, last=torch.zeros((36, 59, 3)),
             structure=torch.zeros((36, 59, 3)),
             context=torch.zeros((36, 59, 9)))
    result, baseline = m.seed_importance(
        net, z, np.full((36, 59, 3), 2.), np.zeros(3), np.ones(3))
    assert baseline == pytest.approx(0)
    assert result.shape == (157, 4)
    np.testing.assert_allclose(result[0], [2, 2, 2, 2])
    np.testing.assert_array_equal(result[1:], 0)
    assert torch.all(z["x"][..., 0] == 1)
    assert torch.all(z["mask"][..., 0] == 1)


def test_importance_rank_and_temporal_stability():
    one = np.arange(157, dtype=float)
    assert m.ranking_order(one)[0] == 156
    assert m.ranking_order(one)[-1] == 0
    samples = np.broadcast_to(one[None, :, None], (10, 157, 4)).copy()
    result = m.rank_stability(samples)
    assert result["ten_seed_mean_pairwise_rank_spearman"] == pytest.approx(1)
    assert result["three_train_block_mean_pairwise_rank_spearman"] == pytest.approx(1)
    assert all(value == pytest.approx(1)
               for value in result["mean_topk_jaccard_vs_ten_seed_rankings"].values())
    names = [f"f{i}" for i in range(157)]
    order = m.ranking_order(one)
    frame = pd.DataFrame(dict(
        feature_rank=range(1, 158),
        original_index=order,
        feature=[names[i] for i in order],
        mean_train_ablation_delta=one[list(order)]))
    assert m.validate_ranking(frame, names) == order
    with pytest.raises(RuntimeError):
        m.validate_ranking(frame.iloc[:-1], names)


def test_all_k_selection_requires_exhaustion_and_prespecified_gates():
    baseline = dict(axis_mae=2, ici_mae=1.7, **{
        f"{year}_mae": 2 for year in (2020, 2021, 2022)})
    better = dict(axis_mae=1.9, ici_mae=1.69,
                  **{"2020_mae": 1.8, "2021_mae": 1.9, "2022_mae": 2.1})
    assert m.qualification(better, baseline)["qualified"] is True
    assert not m.qualification(dict(better, ici_mae=1.71), baseline)["qualified"]
    assert not m.qualification(dict(better, **{
        "2020_mae": 2.1, "2021_mae": 2.2}), baseline)["qualified"]
    records = [dict(K=k, axis_mae=2.0, qualified=False) for k in m.ALL_K]
    assert m.select_k(records) == 157
    records[78].update(axis_mae=1.8, qualified=True)
    records[34].update(axis_mae=1.7, qualified=True)
    assert m.select_k(records) == 35
    with pytest.raises(ValueError):
        m.select_k(records[:-1])


def test_random_controls_unconditional_and_repeatable():
    sets = [m.random_indices(s) for s in m.RANDOM_SUBSET_SEEDS]
    assert len(set(sets)) == len(sets)
    assert all(len(a) == 79 and tuple(sorted(set(a))) == a for a in sets)
    assert sets[0] == m.random_indices(m.RANDOM_SUBSET_SEEDS[0])
    with pytest.raises(ValueError):
        m.random_indices(123)
    order = tuple(range(157))
    assert m.topk_indices(order, 79) == tuple(range(79))
    with pytest.raises(ValueError):
        m.topk_indices(order, 158)


def test_mean_of_ten_individual_seed_errors_not_ensemble_error(
        ctx, tmp_path, monkeypatch):
    truth = ctx["panel"].y[56:68]
    expected_predictions = [truth + float(j - 4.5) for j in range(10)]
    calls = []

    def mock_fit(_ctx, _subset, _data, seed, _device):
        calls.append(seed)
        return dict(
            prediction=expected_predictions[seed - 2026],
            state_dict={"dummy": torch.tensor([seed])},
            parameters=67158, seconds=0.01,
            inference_draws=30, inference_regime="B_MTGNN_MC_DROPOUT")

    monkeypatch.setattr(ctx["m07"], "fit_subset_model", mock_fit)
    indices = tuple(range(156))
    row, interrupted = m.score_policy(
        ctx, tmp_path, indices, "top156", "TOPK",
        "test-contract", "cpu", time.perf_counter(), 10)
    assert interrupted is None
    assert len(calls) == 10
    expected, individual = ctx["m06"].seed_policy_metrics(
        np.stack(expected_predictions), ctx["panel"])
    assert row["axis_mae"] == pytest.approx(expected["axis_mae"])
    assert row["ici_mae"] == pytest.approx(expected["ici_mae"])
    assert row["axis_mae"] > 0
    assert len(row["seed_metrics"]) == 10
    assert len(list((tmp_path / "models" / "top156").glob("*.pt"))) == 10

    def fail_if_retrained(*_a, **_k):
        raise AssertionError("Valid cached seed must not be retrained")

    monkeypatch.setattr(ctx["m07"], "fit_subset_model", fail_if_retrained)
    again, interrupted = m.score_policy(
        ctx, tmp_path, indices, "top156", "TOPK",
        "test-contract", "cpu", time.perf_counter(), 10)
    assert interrupted is None
    assert again["axis_mae"] == pytest.approx(row["axis_mae"])
    with open(tmp_path / "models" / "top156" / "B-MTGNN__s2026.npz",
              "ab") as file:
        file.write(b"tampered")
    with pytest.raises(RuntimeError, match="model cache"):
        m.score_policy(ctx, tmp_path, indices, "top156", "TOPK",
                       "test-contract", "cpu", time.perf_counter(), 10)


def test_resumable_budget_and_frozen_baseline_no_retraining(ctx, tmp_path):
    baseline, interrupted = m.score_policy(
        ctx, tmp_path, tuple(range(157)), "top157",
        "TOPK", "test-contract", "cpu", time.perf_counter(), 10)
    assert interrupted is None
    assert baseline["reused_stage06_baseline"]
    assert baseline["axis_mae"] == pytest.approx(ctx["baseline"]["axis_mae"])
    assert not (tmp_path / "models" / "top157").exists()
    incomplete, next_job = m.score_policy(
        ctx, tmp_path, tuple(range(156)), "top156",
        "TOPK", "test-contract", "cpu", time.perf_counter() - 120, 1)
    assert incomplete is None
    assert next_job["next_seed"] == 2026


def test_true_k_input_initialization_parity_and_real_single_seed(ctx):
    indices = tuple(range(156))
    selected = m.subset(ctx, indices, "probe")
    original = ctx["m07"].instantiate_subset(
        ctx, selected, 2026, compact=False)
    compact = ctx["m07"].instantiate_subset(
        ctx, selected, 2026, compact=True)
    assert compact.base.values.in_features == 156
    assert compact.base.missing.in_features == 156
    assert sum(p.numel() for p in compact.parameters()) == 67158
    torch.testing.assert_close(
        compact.base.values.weight,
        original.base.values.weight[:, list(indices)])
    torch.testing.assert_close(
        compact.base.missing.weight,
        original.base.missing.weight[:, list(indices)])
    full = ctx["m07"].adapt_inputs(ctx["full_fold"], selected, "compact")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted = ctx["m07"].fit_subset_model(ctx, selected, full, 2026, device)
    assert fitted["prediction"].shape == (12, 59, 3)
    assert np.isfinite(fitted["prediction"]).all()
    assert fitted["parameters"] == 67158
    assert fitted["inference_draws"] == 30


def test_pareto_nondominance_is_descriptive():
    records = [
        dict(K=1, axis_mae=2.6, ici_mae=2.0, parameters=100),
        dict(K=2, axis_mae=2.4, ici_mae=1.9, parameters=110),
        dict(K=3, axis_mae=2.5, ici_mae=2.0, parameters=120)]
    table = m.pareto_table(records)
    assert table.nondominated.tolist() == [True, True, False]


def test_completed_train_ranking_reproduces_ten_real_seeds(ctx):
    path = m.RESULTS / "ranking" / "feature_importance.csv"
    if not path.exists():
        pytest.skip("Train-only ranking has not been executed")
    study_sig = m.signature(m.contract(ctx))
    frame, stability, paused = m.build_ranking(
        ctx, m.RESULTS, study_sig, "cpu", time.perf_counter(), .01)
    assert paused is None
    assert len(frame) == 157
    assert m.validate_ranking(frame, ctx["panel"].features)
    assert len(list((m.RESULTS / "ranking" / "seeds").glob("*.npz"))) == 10
    assert np.isfinite(
        stability["ten_seed_mean_pairwise_rank_spearman"])
    assert np.isfinite(
        stability["three_train_block_mean_pairwise_rank_spearman"])
    assert m.read_json(m.RESULTS / "run_manifest.json")["test_opened"] is False
