import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tornado_predictor.build_hourly import BASE_FEATURE_NAMES, build_hourly_run, derive_wholebin_run, hourly_run_path
from tornado_predictor.build_scaled import run_staging_path
from tornado_predictor.convlstm_unet import TornadoConvLSTMUNet
from tornado_predictor.evaluate import compute_metrics, get_proba_labels
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint
from tornado_predictor.sequence_dataset import DenseSequenceDataset, flatten_run_bins
from tornado_predictor.sequence_study import run_list_hash, score_runs, stratified_run_indices, train_and_evaluate
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import QuietMapDownsampler, train_model

GRID = build_coarse_grid()
SHAPE = (GRID.n_rows, GRID.n_cols)
REPO = Path(__file__).resolve().parents[1]
HOURLY = REPO / "data/interim/scaled_build_hourly"
WHOLEBIN = REPO / "data/interim/scaled_build_wholebin"


def _labels(rows):
    df = pd.DataFrame(rows, columns=["event_id", "init_time", "bin_index", "row", "col"])
    return df.astype({"bin_index": "int64", "row": "int64", "col": "int64", "init_time": "datetime64[ns]"})


def _pull(init_time, fxx, grid):
    rng = np.random.default_rng(int(init_time.value % 2**31) + fxx)  # distinct random fields per run and hour
    return {n: rng.normal(size=SHAPE).astype("float32") for n in BASE_FEATURE_NAMES}


@pytest.fixture
def staged(tmp_path):
    """Six tiny synthetic runs; runs 0-2 have a positive cell (varying bin), 3-5 are quiet."""
    hourly, wholebin = tmp_path / "hourly", tmp_path / "wholebin"
    hourly.mkdir()
    inits = [pd.Timestamp("2020-06-01 00:00") + pd.Timedelta(hours=i) for i in range(6)]
    labels = _labels([("e0", inits[0], 0, 10, 10), ("e1", inits[1], 1, 20, 30), ("e2", inits[2], 0, 40, 50), ("e2", inits[2], 1, 40, 50)])
    for init in inits:
        build_hourly_run(init, GRID, None, pull=_pull).to_netcdf(hourly_run_path(hourly, init))
        derive_wholebin_run(hourly_run_path(hourly, init), labels, GRID, run_staging_path(wholebin, init))
    return DenseSequenceDataset(hourly, wholebin, inits)


def test_stratified_indices_keep_the_active_share_and_original_order():
    active = np.array([True] * 60 + [False] * 40)
    idx = stratified_run_indices(active, 20, seed=1)
    assert len(idx) == 20 and idx == sorted(idx)
    assert active[idx].sum() == 12  # 60% of 20
    assert stratified_run_indices(active, 20, seed=1) == idx  # deterministic
    assert stratified_run_indices(active, 500) == list(range(100))
    assert len(stratified_run_indices(np.zeros(10, dtype=bool), 4)) == 4


def test_subset_shares_labels_and_reads_the_right_runs(staged):
    sub = staged.subset([4, 1])
    assert sub.runs == [staged.runs[4], staged.runs[1]]
    np.testing.assert_array_equal(sub.y[1], staged.y[1])
    np.testing.assert_array_equal(sub[0][0], staged[4][0])
    assert sub.is_active.tolist() == [False, True]


def test_run_list_hash_is_order_sensitive_and_stable():
    a, b = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    assert run_list_hash([a, b]) == run_list_hash([a, b]) != run_list_hash([b, a])


def test_score_runs_order_matches_per_run_predictions_and_get_proba_labels(staged):
    torch.manual_seed(0)
    model = TornadoConvLSTMUNet(14, base_channels=4, hidden_channels=8)
    scores = score_runs(model, staged)
    assert scores.shape == (len(staged) * N_BINS, *SHAPE) and scores.dtype == np.float32
    for k in (0, 3):  # sample 2k+b is (run k, bin b)
        x, _ = staged[k]
        with torch.no_grad():
            p = torch.sigmoid(model.eval()(torch.from_numpy(x).unsqueeze(0)))[0].numpy()
        np.testing.assert_allclose(scores[2 * k], p[0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(scores[2 * k + 1], p[1], rtol=1e-5, atol=1e-7)
    y_score, y_true = get_proba_labels(model, staged)  # the existing evaluation path, unchanged
    np.testing.assert_allclose(y_score, scores.ravel(), rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(y_true, flatten_run_bins(staged.y).ravel())


def test_train_model_is_deterministic_and_loss_falls_on_a_sequence_dataset(staged):
    def run(seed):
        torch.manual_seed(seed)
        m = TornadoConvLSTMUNet(14, base_channels=4, hidden_channels=8)
        r = train_model(staged, m, epochs=6, batch_size=1, seed=seed, quiet_keep_fraction=1.0, lr=3e-3)
        return r["loss_history"], r["epoch_seconds"]

    (a, secs), (b, _) = run(0), run(0)
    assert a == b
    assert len(secs) == 6 and all(s > 0 for s in secs)
    assert a[-1] < a[0]


def test_quiet_downsampler_uses_run_level_flags_for_the_sequence_dataset(staged):
    kept = list(QuietMapDownsampler(staged.is_active, quiet_keep_fraction=0.0, seed=0))
    assert sorted(kept) == [0, 1, 2]


def test_ema_tracking_does_not_change_training_on_a_sequence_dataset(staged):
    def run(ema):
        torch.manual_seed(0)
        m = TornadoConvLSTMUNet(14, base_channels=4, hidden_channels=8)
        r = train_model(staged, m, epochs=2, batch_size=1, seed=0, quiet_keep_fraction=1.0, ema_decays=ema)
        return r, m

    (r0, m0), (r1, m1) = run(()), run((0.9,))
    assert r0["loss_history"] == r1["loss_history"]
    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(p0, p1)
    assert "ema_state_dicts" in r1


def test_train_and_evaluate_end_to_end_checkpoints_reload_and_reproduce_scores(staged, tmp_path):
    train_ds, test_ds = staged.subset([0, 1, 2, 3]), staged.subset([4, 5, 2])
    res = train_and_evaluate(train_ds, test_ds, out_dir=tmp_path / "out", held_out_year=2020, seed=3, epochs=1,
                             base_channels=4, hidden_channels=8, ema_decays=(0.9,))
    assert set(res["checkpoints"]) == {"raw", "ema0.9"} and res["n_train_runs"] == 4 and res["n_test_runs"] == 3
    assert res["n_test_positive_cells"] == 2 and len(res["epoch_seconds"]) == 1
    assert res["val_over_train_loss"] > 0 and np.isfinite(res["metrics"]["raw"]["auc_pr"])
    json.dumps(res)  # results are JSON-serialisable as they stand
    for name in ("raw", "ema0.9"):
        model, ckpt = load_checkpoint(res["checkpoints"][name])
        assert isinstance(model, TornadoConvLSTMUNet) and ckpt["model_class"] == "TornadoConvLSTMUNet"
        assert_feature_order_matches(test_ds.feature_names, ckpt)
        saved = np.load(res["scores"][name])
        assert saved.shape == (6, *SHAPE) and saved.dtype == np.float16
        np.testing.assert_allclose(score_runs(model, test_ds), saved.astype(np.float32), atol=1e-3)
    assert "ema_decay" in torch.load(res["checkpoints"]["ema0.9"], weights_only=False)


@pytest.mark.skipif(not (HOURLY.exists() and WHOLEBIN.exists()), reason="staging not built")
def test_cli_end_to_end_on_a_tiny_real_subset(tmp_path):
    out = tmp_path / "snippet"
    cmd = [sys.executable, str(REPO / "scripts/11_convlstm/train_convlstm.py"), "--fold", "2020", "--seed", "10", "--epochs", "1",
           "--max-train-runs", "4", "--base-channels", "4", "--hidden-channels", "8", "--ema-decays", "0.9", "--out-dir", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr[-2000:]
    res = json.loads((out / "results.json").read_text())
    assert res["n_train_runs"] == 4 and res["n_test_runs"] == 438
    assert res["test_order_matches_arm_a_scores"] is True
    runs = json.loads((out / "train_runs.json").read_text())
    assert runs["hash"] == res["train_run_list_hash"] and len(runs["runs"]) == 4
    assert np.load(res["scores"]["raw"]).shape == (876, *SHAPE)


def test_grad_clip_norm_limits_the_update_and_is_recorded(staged):
    def weights_after(clip):
        torch.manual_seed(0)
        m = TornadoConvLSTMUNet(14, base_channels=4, hidden_channels=8)
        before = [p.detach().clone() for p in m.parameters()]
        r = train_model(staged, m, epochs=1, batch_size=1, seed=0, quiet_keep_fraction=1.0, grad_clip_norm=clip)
        moved = sum((p.detach() - b).norm() ** 2 for p, b in zip(m.parameters(), before)).sqrt().item()
        return moved, r["hyperparameters"]
    free, hp_free = weights_after(None)
    tiny, hp_tiny = weights_after(1e-9)
    assert tiny < free and "grad_clip_norm" not in hp_free and hp_tiny["grad_clip_norm"] == 1e-9


def test_repair_options_flow_into_the_saved_checkpoint(staged, tmp_path):
    res = train_and_evaluate(staged.subset([0, 1, 2, 3]), staged.subset([4, 5, 2]), out_dir=tmp_path / "o", held_out_year=2020, seed=3, epochs=1,
                             base_channels=8, hidden_channels=8, ema_decays=(), norm_bottleneck=True, pooled_skips=True, grad_clip_norm=1.0)
    assert res["model_hyperparameters"]["norm_bottleneck"] and res["model_hyperparameters"]["pooled_skips"] and res["grad_clip_norm"] == 1.0
    model, _ = load_checkpoint(res["checkpoints"]["raw"])
    assert model.norm_bottleneck is not None and model.pooled_skips


def test_per_epoch_metrics_flow_through_train_and_evaluate_and_match_final_metrics(staged, tmp_path):
    kw = dict(held_out_year=2020, seed=3, epochs=2, base_channels=4, hidden_channels=8, ema_decays=(0.9,))
    plain = train_and_evaluate(staged.subset([0, 1, 2, 3]), staged.subset([4, 5, 2]), out_dir=tmp_path / "a", **kw)
    res = train_and_evaluate(staged.subset([0, 1, 2, 3]), staged.subset([4, 5, 2]), out_dir=tmp_path / "b", per_epoch_metrics=True, **kw)
    assert "epoch_metrics" not in plain
    assert set(res["epoch_metrics"]) == {"raw", "ema0.9"} and all(len(v) == 2 for v in res["epoch_metrics"].values())
    json.dumps(res)
    # the last epoch's per-epoch metrics are the same numbers the final scoring reports (same weights, same held-out set)
    for name, key in (("raw", "raw"), ("ema0.9", "ema0.9")):
        assert res["epoch_metrics"][key][-1]["auc_pr"] == pytest.approx(res["metrics"][name]["auc_pr"], rel=1e-9)
    # and turning the hook on does not change training
    assert res["loss_history"] == plain["final_train_loss"] or res["loss_history"][-1] == plain["final_train_loss"]


def test_early_stopping_flows_through_train_and_evaluate(staged, tmp_path):
    kw = dict(held_out_year=2020, seed=3, base_channels=4, hidden_channels=8, ema_decays=(0.9,))
    train, test, val = staged.subset([0, 1, 2, 3]), staged.subset([4, 5, 2]), staged.subset([5, 1])
    plain = train_and_evaluate(train, test, out_dir=tmp_path / "a", epochs=3, **kw)
    res = train_and_evaluate(train, test, out_dir=tmp_path / "b", epochs=3, val_ds=val, patience=1, min_epochs=1, **kw)
    assert "early_stopping" not in plain and "n_val_runs" not in plain
    es = res["early_stopping"]
    json.dumps(res)
    assert res["n_val_runs"] == 2 and es["monitor"] == "ema0.9" and 1 <= es["best_epoch"] <= es["stopped_epoch"] <= 3
    assert len(es["history"]) == es["stopped_epoch"] == len(res["epoch_seconds"])
    # the checkpoint carries the best-epoch EMA weights: scoring it reproduces the reported test metric
    model, _ = load_checkpoint(res["checkpoints"]["ema0.9"])
    assert np.isclose(compute_metrics(flatten_run_bins(test.y).ravel(), score_runs(model, test).ravel(), include_curves=False)["auc_pr"], res["metrics"]["ema0.9"]["auc_pr"])
    if es["stopped_epoch"] == 3 and es["best_epoch"] == 3:  # never triggered: identical to the run without early stopping
        assert res["metrics"]["ema0.9"]["auc_pr"] == plain["metrics"]["ema0.9"]["auc_pr"]
