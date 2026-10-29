"""
Training, scoring and checkpointing for the ConvLSTM study (docs/convlstm_spec.md, T3).

No new training loop: `training.train_model` already works for a run-level sequence dataset
(it squeezes dim 1 of the logits, a no-op for `(B, 2, row, col)`; FocalLoss averages over
both bins; `QuietMapDownsampler` reads the dataset's run-level `is_active`; EMA is generic).
This module adds only what is specific to the study:

* `stratified_run_indices` -- an optional proportional subset for dev-scale runs (keeps the
  active/quiet mix); a fallback, since a full-fold epoch may already fit the snippet budget.
* `score_runs` -- per-cell probabilities in ARM A's `(run, bin)` sample order, so the existing
  evaluation (pooled metrics, map-level diagnostics, cached-score ensembling) applies as is.
* `train_and_evaluate` -- one fold, one seed: train, save raw + EMA checkpoints (self-describing,
  loadable through `inference.load_checkpoint`), score the held-out set once per model, write
  float16 scores and a results dict.

Not supported: `cell_mask` (`masked_is_active` assumes 3-D labels); not needed for this study.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tornado_predictor.convlstm_unet import TornadoConvLSTMUNet
from tornado_predictor.evaluate import compute_metrics
from tornado_predictor.sequence_dataset import DenseSequenceDataset, flatten_run_bins
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import train_model

MODEL_CLASS = "TornadoConvLSTMUNet"


def run_list_hash(runs: list) -> str:
    """Short, order-sensitive hash of a run list, so two studies can prove they used the same runs."""
    text = "\n".join(pd.Timestamp(r).isoformat() for r in runs)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stratified_run_indices(is_active: np.ndarray, n_runs: int, seed: int = 0) -> list[int]:
    """`n_runs` positions chosen at random WITHOUT changing the active/quiet proportion
    (active and quiet each sampled at the same rate), returned in original order. n_runs >=
    len(is_active) returns everything."""
    is_active = np.asarray(is_active, dtype=bool)
    n = len(is_active)
    if n_runs >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    active, quiet = np.where(is_active)[0], np.where(~is_active)[0]
    n_active = int(round(n_runs * len(active) / n))
    n_active = min(n_active, len(active), n_runs)
    n_quiet = min(n_runs - n_active, len(quiet))
    picked = np.concatenate([rng.choice(active, n_active, replace=False), rng.choice(quiet, n_quiet, replace=False)])
    return sorted(int(i) for i in picked)


def score_runs(model: torch.nn.Module, dataset: DenseSequenceDataset) -> np.ndarray:
    """Probabilities `(n_runs * N_BINS, row, col)` float32 in arm A's `(run, bin)` sample order."""
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            out.append(torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0)))[0].numpy())  # (N_BINS, row, col)
    model.train(was_training)
    return flatten_run_bins(np.stack(out))


def _checkpoint(model_state, hyper, feature_names, train_result, extra=None) -> dict:
    return {
        "model_class": MODEL_CLASS,
        "model_state_dict": model_state,
        "model_hyperparameters": hyper,
        "feature_names": list(feature_names),
        "training_hyperparameters": train_result["hyperparameters"],
        **(extra or {}),
    }


def train_and_evaluate(
    train_ds: DenseSequenceDataset,
    test_ds: DenseSequenceDataset,
    *,
    out_dir: Path,
    held_out_year: int,
    seed: int,
    epochs: int,
    batch_size: int = 1,
    base_channels: int = 16,
    hidden_channels: int = 64,
    ema_decays: tuple[float, ...] = (0.9999,),
    alpha: float = 0.25,
    gamma: float = 2.0,
    quiet_keep_fraction: float = 0.2,
    lr: float = 1e-3,
) -> dict:
    """Trains one ConvLSTM-U-Net on `train_ds`, tracking loss on `test_ds` each epoch (the held-out
    year, as every year-holdout study here does), saves raw and EMA checkpoints plus float16
    held-out scores under `out_dir`, and returns a results dict. Defaults mirror arm A's recipe
    except `batch_size=1` run (2 bin-maps per step, matching A's batch of 2 maps)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyper = {
        "in_channels": len(train_ds.feature_names),
        "base_channels": base_channels,
        "hidden_channels": hidden_channels,
        "stop_steps": [3, 7],
    }
    t0 = time.time()
    torch.manual_seed(seed)  # weight init; train_model seeds its own sampler
    model = TornadoConvLSTMUNet(**hyper)
    result = train_model(
        train_ds, model, val_dataset=test_ds, epochs=epochs, batch_size=batch_size, lr=lr, alpha=alpha, gamma=gamma,
        quiet_keep_fraction=quiet_keep_fraction, seed=seed, ema_decays=tuple(ema_decays),
    )
    train_seconds = time.time() - t0

    meta = {"loss_history": result["loss_history"], "val_loss_history": result["val_loss_history"]}
    y_true = flatten_run_bins(test_ds.y)
    variants = {"raw": model}
    for decay, sd in result.get("ema_state_dicts", {}).items():
        ema = TornadoConvLSTMUNet(**hyper)
        ema.load_state_dict(sd)
        variants[f"ema{decay}"] = ema

    metrics, ckpts, score_paths = {}, {}, {}
    for name, m in variants.items():
        scores = score_runs(m, test_ds)
        metrics[name] = compute_metrics(y_true.ravel(), scores.ravel(), include_curves=False)
        stem = f"convlstm_heldout_{held_out_year}_s{seed}" + ("" if name == "raw" else f"_{name}")
        score_paths[name] = out_dir / f"{stem}_scores.npy"
        np.save(score_paths[name], scores.astype(np.float16))
        ckpts[name] = out_dir / f"{stem}.pt"
        torch.save(
            _checkpoint(m.state_dict(), hyper, train_ds.feature_names, result,
                        {**meta, **({"ema_decay": float(name[3:])} if name != "raw" else {})}),
            ckpts[name],
        )

    return {
        "held_out_year": held_out_year,
        "seed": seed,
        "model_class": MODEL_CLASS,
        "model_hyperparameters": hyper,
        "n_train_runs": len(train_ds),
        "n_train_active_runs": int(train_ds.is_active.sum()),
        "n_test_runs": len(test_ds),
        "n_test_active_runs": int(test_ds.is_active.sum()),
        "n_test_positive_cells": int(test_ds.y.sum()),
        "train_run_list_hash": run_list_hash(train_ds.runs),
        "train_seconds": train_seconds,
        "epoch_seconds": result["epoch_seconds"],
        "final_train_loss": result["loss_history"][-1],
        "final_val_loss": result["val_loss_history"][-1],
        "val_over_train_loss": result["val_loss_history"][-1] / result["loss_history"][-1],
        "training_hyperparameters": result["hyperparameters"],
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "scores": {k: str(v) for k, v in score_paths.items()},
        "metrics": metrics,
    }
