"""
Sweep TornadoCNN's width (hidden_channels) and depth (n_hidden_layers)
on the full-scale (v2) dataset, at the project's current best-known
hyperparameters otherwise (alpha=0.25, gamma=2.0, lr=1e-3, batch_size=2,
quiet_keep_fraction=0.2, epochs=50, seed=0) -- the cheapest, lowest-risk
architecture experiment: stay inside the current full-resolution,
no-pooling design, just make it bigger, per model.py's own docstring
note that this was always the flagged next iteration.

Config set (4 new variants + the existing baseline, same size as the
alpha sweep): (64, 3) and (128, 3) isolate channel-width alone; (32, 5)
isolates receptive-field/depth alone (n_hidden_layers=5 -> 11x11 cell
receptive field, ~430km, vs. the baseline's 7x7/~270km); (128, 5) is
the "go big on both axes" combined variant. Deliberately skips a full
2x2 grid to keep this tractable and interpretable -- a targeted
ablation, not a blind grid search.

The existing baseline checkpoint (models/tornado_cnn_full.pt) is
re-evaluated via evaluate.evaluate_checkpoint (not retrained) and
included as the first results row, so the comparison point comes from
the exact same code path as the new variants rather than a hardcoded
prior number -- ticket 1 already proved this reproduces AUC-PR=0.0283
exactly.

Usage:
    python scripts/sweep_architecture.py
    python scripts/sweep_architecture.py --configs 64,3 128,3 32,5 128,5
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
DEFAULT_BASELINE_CHECKPOINT = REPO_ROOT / "models" / "tornado_cnn_full.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "architecture_sweep"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

DEFAULT_CONFIGS = [(64, 3), (128, 3), (32, 5), (128, 5)]

ALPHA = 0.25
GAMMA = 2.0
LR = 1e-3
BATCH_SIZE = 2
QUIET_KEEP_FRACTION = 0.2
EPOCHS = 50
SEED = 0


def parse_configs(raw: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in pair.split(",")) for pair in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument(
        "--configs", type=str, nargs="+", default=[f"{h},{n}" for h, n in DEFAULT_CONFIGS],
        help="hidden_channels,n_hidden_layers pairs, e.g. 64,3 128,5",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DenseGridDataset(str(args.train))
    val_ds = DenseGridDataset(str(args.val))
    print(f"train: {len(train_ds)} samples ({int(train_ds.is_active.sum())} active) | "
          f"val: {len(val_ds)} samples ({int(val_ds.is_active.sum())} active) | "
          f"{len(train_ds.feature_names)} features")

    results = []

    print("\n=== baseline (re-evaluated, not retrained) ===")
    baseline_metrics = evaluate_checkpoint(str(args.baseline_checkpoint), str(args.val))
    results.append({
        "config": "baseline",
        "hidden_channels": 32,
        "n_hidden_layers": 3,
        "n_params": None,
        "receptive_field_cells": receptive_field_cells(3),
        "checkpoint": str(args.baseline_checkpoint),
        **baseline_metrics,
    })
    print(f"AUC-PR={baseline_metrics['auc_pr']:.4f} AUC-ROC={baseline_metrics['auc_roc']:.4f} "
          f"cohens_d={baseline_metrics['cohens_d']:.2f}")
    args.results.write_text(json.dumps(results, indent=2))

    for hidden_channels, n_hidden_layers in configs:
        label = f"h{hidden_channels}_l{n_hidden_layers}"
        rf = receptive_field_cells(n_hidden_layers)
        print(f"\n=== {label} (receptive field {rf}x{rf} cells) ===")
        t0 = time.time()

        torch.manual_seed(SEED)
        model = TornadoCNN(in_channels=len(train_ds.feature_names), hidden_channels=hidden_channels, n_hidden_layers=n_hidden_layers)
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
                "hidden_channels": hidden_channels,
                "n_hidden_layers": n_hidden_layers,
            },
            "feature_names": train_ds.feature_names,
        }, ckpt_path)

        results.append({
            "config": label,
            "hidden_channels": hidden_channels,
            "n_hidden_layers": n_hidden_layers,
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
