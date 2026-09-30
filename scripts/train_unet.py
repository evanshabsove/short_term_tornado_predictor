"""
Trains TornadoUNet (unet.py) on the full-scale (v2) dataset, at the
project's current best-known hyperparameters otherwise (alpha=0.25,
gamma=2.0, lr=1e-3, batch_size=2, quiet_keep_fraction=0.2, epochs=50,
seed=0). A single training run, not a sweep -- unlike tickets 2-3
(width/depth and dilation ablations), this ticket builds and trains one
U-Net configuration (base_channels=16, deliberately conservative given
ticket 2's overfitting lesson at higher parameter counts -- see
unet.py's module docstring).

Also re-evaluates the current best checkpoint
(models/dilation_sweep/tornado_cnn_d1-2-4-8.pt, AUC-PR 0.0322) via
evaluate.evaluate_checkpoint (not retrained) as the first results row,
same "reference point from the identical code path" practice as
tickets 2-3.

Usage:
    python scripts/train_unet.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tornado_predictor.evaluate import evaluate_checkpoint, evaluate_model
from tornado_predictor.training import DenseGridDataset, train_model
from tornado_predictor.unet import TornadoUNet

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_train_full.nc"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"
DEFAULT_REFERENCE_CHECKPOINT = REPO_ROOT / "models" / "dilation_sweep" / "tornado_cnn_d1-2-4-8.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "unet"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

BASE_CHANNELS = 16

ALPHA = 0.25
GAMMA = 2.0
LR = 1e-3
BATCH_SIZE = 2
QUIET_KEEP_FRACTION = 0.2
EPOCHS = 50
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DenseGridDataset(str(args.train))
    val_ds = DenseGridDataset(str(args.val))
    print(f"train: {len(train_ds)} samples ({int(train_ds.is_active.sum())} active) | "
          f"val: {len(val_ds)} samples ({int(val_ds.is_active.sum())} active) | "
          f"{len(train_ds.feature_names)} features")

    results = []

    print(f"\n=== reference: {args.reference_checkpoint.name} (re-evaluated, not retrained) ===")
    ref_metrics = evaluate_checkpoint(str(args.reference_checkpoint), str(args.val))
    results.append({
        "config": "reference (dilated d1-2-4-8)",
        "n_params": None,
        "checkpoint": str(args.reference_checkpoint),
        **ref_metrics,
    })
    print(f"AUC-PR={ref_metrics['auc_pr']:.4f} AUC-ROC={ref_metrics['auc_roc']:.4f} cohens_d={ref_metrics['cohens_d']:.2f}")
    args.results.write_text(json.dumps(results, indent=2))

    print(f"\n=== TornadoUNet (base_channels={args.base_channels}) ===")
    t0 = time.time()

    torch.manual_seed(SEED)
    model = TornadoUNet(in_channels=len(train_ds.feature_names), base_channels=args.base_channels)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params={n_params:,}")

    train_result = train_model(
        train_ds, model, val_dataset=val_ds, epochs=args.epochs, batch_size=BATCH_SIZE,
        lr=LR, alpha=ALPHA, gamma=GAMMA, quiet_keep_fraction=QUIET_KEEP_FRACTION, seed=SEED,
    )
    metrics = evaluate_model(model, val_ds)
    elapsed = time.time() - t0

    print(f"final_train_loss={train_result['loss_history'][-1]:.5f} "
          f"final_val_loss={train_result['val_loss_history'][-1]:.5f}")
    print(f"AUC-PR={metrics['auc_pr']:.4f} AUC-ROC={metrics['auc_roc']:.4f} "
          f"cohens_d={metrics['cohens_d']:.2f}  ({elapsed:.0f}s)")

    ckpt_path = args.out_dir / "tornado_unet.pt"
    torch.save({
        "model_class": "TornadoUNet",
        "model_state_dict": model.state_dict(),
        "loss_history": train_result["loss_history"],
        "val_loss_history": train_result["val_loss_history"],
        "training_hyperparameters": train_result["hyperparameters"],
        "model_hyperparameters": {
            "in_channels": len(train_ds.feature_names),
            "base_channels": args.base_channels,
        },
        "feature_names": train_ds.feature_names,
    }, ckpt_path)

    results.append({
        "config": f"unet_base{args.base_channels}",
        "n_params": n_params,
        "elapsed_sec": elapsed,
        "final_train_loss": train_result["loss_history"][-1],
        "final_val_loss": train_result["val_loss_history"][-1],
        "loss_history": train_result["loss_history"],
        "val_loss_history": train_result["val_loss_history"],
        "checkpoint": str(ckpt_path),
        **metrics,
    })
    args.results.write_text(json.dumps(results, indent=2))

    print(f"\nSaved results to {args.results}")


if __name__ == "__main__":
    main()
