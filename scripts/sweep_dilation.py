"""
Sweep TornadoCNN's per-layer dilation schedule on the full-scale (v2)
dataset, at the project's current best-known hyperparameters otherwise
(alpha=0.25, gamma=2.0, lr=1e-3, batch_size=2, quiet_keep_fraction=0.2,
epochs=50, seed=0). Channel width is fixed at 32 throughout (ticket 2's
best-performing, cheapest width -- ticket 2 already showed widening
channels alone doesn't help, so this isolates the dilation effect
instead of re-testing width).

Motivation: ticket 2 found that growing receptive field via more plain
3x3 layers helped (32ch/5L, 11x11 cells/~430km, beat the baseline's
AUC-PR). Dilated convolutions grow receptive field much faster per
layer than stacking more plain layers, without adding pooling or a
U-Net's grid-alignment complexity. Receptive field =
1 + 2*sum(dilations) (model.receptive_field_cells) -- e.g. dilations
[1,2,4,8] reach 31 cells (~1209km), computed exactly, not the "hundreds
of km" the ticket was originally framed around; [1,2,4] (585km) is
tested alongside it as the more conservative, literally-"hundreds of
km" schedule.

Two existing checkpoints are re-evaluated (not retrained) as reference
points: the original baseline (models/tornado_cnn_full.pt, AUC-PR
0.0283) and ticket 2's winner
(models/architecture_sweep/tornado_cnn_h32_l5.pt, AUC-PR 0.0319).

Usage:
    python scripts/sweep_dilation.py
    python scripts/sweep_dilation.py --dilations 1,2,4 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tornado_predictor.evaluate import evaluate_checkpoint, evaluate_model
from tornado_predictor.model import TornadoCNN, receptive_field_cells
from tornado_predictor.training import DenseGridDataset, train_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_train_full.nc"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"
DEFAULT_REFERENCE_CHECKPOINTS = {
    "baseline (32ch, 3L, no dilation)": REPO_ROOT / "models" / "tornado_cnn_full.pt",
    "ticket2 winner (32ch, 5L, no dilation)": REPO_ROOT / "models" / "architecture_sweep" / "tornado_cnn_h32_l5.pt",
}
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "dilation_sweep"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

HIDDEN_CHANNELS = 32
DEFAULT_DILATION_SCHEDULES = [[1, 2, 4], [1, 2, 4, 8]]

ALPHA = 0.25
GAMMA = 2.0
LR = 1e-3
BATCH_SIZE = 2
QUIET_KEEP_FRACTION = 0.2
EPOCHS = 50
SEED = 0


def parse_dilation_schedules(raw: list[str]) -> list[list[int]]:
    return [[int(x) for x in schedule.split(",")] for schedule in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument(
        "--dilations", type=str, nargs="+", default=[",".join(str(d) for d in s) for s in DEFAULT_DILATION_SCHEDULES],
        help="comma-separated per-layer dilation schedules, e.g. 1,2,4 1,2,4,8",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    schedules = parse_dilation_schedules(args.dilations)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DenseGridDataset(str(args.train))
    val_ds = DenseGridDataset(str(args.val))
    print(f"train: {len(train_ds)} samples ({int(train_ds.is_active.sum())} active) | "
          f"val: {len(val_ds)} samples ({int(val_ds.is_active.sum())} active) | "
          f"{len(train_ds.feature_names)} features")

    results = []

    for label, ckpt_path in DEFAULT_REFERENCE_CHECKPOINTS.items():
        print(f"\n=== {label} (re-evaluated, not retrained) ===")
        metrics = evaluate_checkpoint(str(ckpt_path), str(args.val))
        n_layers = 3 if "3L" in label else 5
        results.append({
            "config": label,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_hidden_layers": n_layers,
            "dilations": [1] * n_layers,
            "n_params": None,
            "receptive_field_cells": receptive_field_cells(n_layers),
            "checkpoint": str(ckpt_path),
            **metrics,
        })
        print(f"AUC-PR={metrics['auc_pr']:.4f} AUC-ROC={metrics['auc_roc']:.4f} cohens_d={metrics['cohens_d']:.2f}")
        args.results.write_text(json.dumps(results, indent=2))

    for dilations in schedules:
        label = "d" + "-".join(str(d) for d in dilations)
        rf = receptive_field_cells(len(dilations), dilations=dilations)
        km = rf * 39
        print(f"\n=== {label} (receptive field {rf}x{rf} cells, ~{km}km) ===")
        t0 = time.time()

        torch.manual_seed(SEED)
        model = TornadoCNN(
            in_channels=len(train_ds.feature_names), hidden_channels=HIDDEN_CHANNELS,
            n_hidden_layers=len(dilations), dilations=dilations,
        )
        n_params = sum(p.numel() for p in model.parameters())

        train_result = train_model(
            train_ds, model, val_dataset=val_ds, epochs=args.epochs, batch_size=BATCH_SIZE,
            lr=LR, alpha=ALPHA, gamma=GAMMA, quiet_keep_fraction=QUIET_KEEP_FRACTION, seed=SEED,
        )
        metrics = evaluate_model(model, val_ds)
        elapsed = time.time() - t0

        print(f"n_params={n_params:,} final_train_loss={train_result['loss_history'][-1]:.5f} "
              f"final_val_loss={train_result['val_loss_history'][-1]:.5f}")
        print(f"AUC-PR={metrics['auc_pr']:.4f} AUC-ROC={metrics['auc_roc']:.4f} "
              f"cohens_d={metrics['cohens_d']:.2f}  ({elapsed:.0f}s)")

        ckpt_path = args.out_dir / f"tornado_cnn_{label}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "loss_history": train_result["loss_history"],
            "val_loss_history": train_result["val_loss_history"],
            "training_hyperparameters": train_result["hyperparameters"],
            "model_hyperparameters": {
                "in_channels": len(train_ds.feature_names),
                "hidden_channels": HIDDEN_CHANNELS,
                "n_hidden_layers": len(dilations),
                "dilations": dilations,
            },
            "feature_names": train_ds.feature_names,
        }, ckpt_path)

        results.append({
            "config": label,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_hidden_layers": len(dilations),
            "dilations": dilations,
            "n_params": n_params,
            "receptive_field_cells": rf,
            "elapsed_sec": elapsed,
            "final_train_loss": train_result["loss_history"][-1],
            "final_val_loss": train_result["val_loss_history"][-1],
            "loss_history": train_result["loss_history"],
            "val_loss_history": train_result["val_loss_history"],
            "checkpoint": str(ckpt_path),
            **metrics,
        })
        args.results.write_text(json.dumps(results, indent=2))

    print(f"\nSaved sweep results to {args.results}")


if __name__ == "__main__":
    main()
