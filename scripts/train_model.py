"""
Train the baseline TornadoCNN (model.py) on a consolidated dataset
(dataset.py) using FocalLoss + QuietMapDownsampler (training.py).

Trains on the pilot dataset by default (6 samples, all "active" --
QuietMapDownsampler is a no-op at this scale, see training.py). This
is a mechanics check, not a real model: 6 samples is far too little to
generalize, and this script does not hold out a validation set (a
formal train/val split isn't meaningful at this scale) -- it trains on
all pilot samples and reports training loss only.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --dataset data/processed/training_dataset_pilot.nc --epochs 100 --out models/tornado_cnn_pilot.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, train_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"
DEFAULT_OUT_PATH = REPO_ROOT / "models" / "tornado_cnn_pilot.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--quiet-keep-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    dataset = DenseGridDataset(str(args.dataset))
    n_active, n_quiet = int(dataset.is_active.sum()), int((~dataset.is_active).sum())
    print(f"Loaded dataset: {len(dataset)} samples, {len(dataset.feature_names)} features, "
          f"{n_active} active / {n_quiet} quiet")
    if n_quiet == 0:
        print("(no quiet samples -- QuietMapDownsampler is a no-op on this dataset)")

    # Seeded before model construction so weight init is reproducible
    # too, not just the sampler's own independently-seeded randomness.
    torch.manual_seed(args.seed)
    model = TornadoCNN(in_channels=len(dataset.feature_names), hidden_channels=args.hidden_channels)

    result = train_model(
        dataset,
        model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        quiet_keep_fraction=args.quiet_keep_fraction,
        seed=args.seed,
    )
    loss_history = result["loss_history"]

    print_every = max(1, args.epochs // 10)
    for epoch, loss in enumerate(loss_history):
        if epoch % print_every == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}: loss={loss:.4f}")

    print(f"\nFinal loss: {loss_history[-1]:.4f} (started at {loss_history[0]:.4f})")
    if loss_history[-1] >= loss_history[0]:
        print("WARNING: loss did not decrease -- check learning rate / data / gradients")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "loss_history": loss_history,
        "training_hyperparameters": result["hyperparameters"],
        "model_hyperparameters": {
            "in_channels": len(dataset.feature_names),
            "hidden_channels": args.hidden_channels,
        },
        "feature_names": dataset.feature_names,
    }
    torch.save(checkpoint, args.out)
    print(f"\nSaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
