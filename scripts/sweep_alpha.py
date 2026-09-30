"""
Sweep FocalLoss's alpha hyperparameter on the full-scale (v2) dataset,
training one TornadoCNN per value (all other hyperparameters held at
project defaults: gamma=2.0, lr=1e-3, batch_size=2,
quiet_keep_fraction=0.2, epochs=50, seed=0) and evaluating each
checkpoint's held-out val-set positive/negative probability separation.

This collects real data to settle a direction question this project's
own earlier documentation left ambiguous: CLAUDE.md's "Inference +
sanity check" section says modest positive-cell confidence "traces
directly to alpha=0.25 ... down-weighting the positive class" and
"confirms FocalLoss's own docstring note ('expect to need a lower
alpha')" -- but per training.FocalLoss's actual formula
(alpha_t = alpha for the positive class, (1 - alpha) for negative),
LOWERING alpha below 0.25 would down-weight the rare positive class
even further, not less. That docstring note looks backwards. Rather
than trust either the old comment or intuition, this sweep tries both
directions (0.1 and 0.25 below the midpoint, 0.5/0.75/0.9 above it)
and lets the actual held-out separation numbers decide.

Each run's loss curves and held-out separation stats (pooled and
per-sample) are saved to --results (default
models/alpha_sweep/results.json) incrementally after each alpha
completes, and each trained model to
models/alpha_sweep/tornado_cnn_alpha_<alpha>.pt -- for a follow-up
notebook to load and analyze/visualize.

Usage:
    python scripts/sweep_alpha.py
    python scripts/sweep_alpha.py --alphas 0.1 0.25 0.5 0.75 0.9
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tornado_predictor.inference import predict_proba
from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, train_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_train_full.nc"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "alpha_sweep"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

DEFAULT_ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]


def compute_val_separation(model: TornadoCNN, val_dataset: DenseGridDataset) -> dict:
    """Pooled and per-sample positive/negative probability separation
    across the entire held-out val set -- same computation used to
    evaluate the full-scale training run in the prior session, now
    packaged for reuse across every alpha value."""
    all_pos, all_neg = [], []
    per_sample_ratios = []
    n_misses = 0
    active_idx = np.where(val_dataset.is_active)[0]

    for i in range(len(val_dataset)):
        X, y = val_dataset[i]
        proba = predict_proba(model, X)
        pos_mask = y > 0
        all_pos.append(proba[pos_mask])
        all_neg.append(proba[~pos_mask])

    for i in active_idx:
        X, y = val_dataset[i]
        proba = predict_proba(model, X)
        pos_mask = y > 0
        mean_pos = proba[pos_mask].mean()
        min_pos = proba[pos_mask].min()
        mean_neg = proba[~pos_mask].mean()
        if mean_neg > 0:
            per_sample_ratios.append(float(mean_pos / mean_neg))
        if min_pos < mean_neg:
            n_misses += 1

    all_pos = np.concatenate(all_pos)
    all_neg = np.concatenate(all_neg)
    pooled_std = np.sqrt((all_pos.std() ** 2 + all_neg.std() ** 2) / 2)
    cohens_d = float((all_pos.mean() - all_neg.mean()) / pooled_std) if pooled_std > 0 else float("nan")
    ratios = np.array(per_sample_ratios)

    return {
        "n_positive_cells": int(len(all_pos)),
        "n_negative_cells": int(len(all_neg)),
        "mean_proba_positive": float(all_pos.mean()),
        "mean_proba_negative": float(all_neg.mean()),
        "pooled_ratio": float(all_pos.mean() / all_neg.mean()) if all_neg.mean() > 0 else float("nan"),
        "cohens_d": cohens_d,
        "n_active_val_samples": int(len(active_idx)),
        "n_misses": int(n_misses),
        "miss_rate": float(n_misses / len(active_idx)) if len(active_idx) else float("nan"),
        "per_sample_ratio_mean": float(ratios.mean()) if len(ratios) else float("nan"),
        "per_sample_ratio_median": float(np.median(ratios)) if len(ratios) else float("nan"),
        "per_sample_ratio_p5": float(np.percentile(ratios, 5)) if len(ratios) else float("nan"),
        "per_sample_ratio_p25": float(np.percentile(ratios, 25)) if len(ratios) else float("nan"),
        "per_sample_ratio_p75": float(np.percentile(ratios, 75)) if len(ratios) else float("nan"),
        "per_sample_ratio_p95": float(np.percentile(ratios, 95)) if len(ratios) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quiet-keep-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
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
    for alpha in args.alphas:
        print(f"\n=== alpha={alpha} ===")
        t0 = time.time()

        torch.manual_seed(args.seed)
        model = TornadoCNN(in_channels=len(train_ds.feature_names))

        train_result = train_model(
            train_ds,
            model,
            val_dataset=val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=alpha,
            gamma=args.gamma,
            quiet_keep_fraction=args.quiet_keep_fraction,
            seed=args.seed,
        )
        sep = compute_val_separation(model, val_ds)
        elapsed = time.time() - t0

        print(f"final train_loss={train_result['loss_history'][-1]:.4f} "
              f"val_loss={train_result['val_loss_history'][-1]:.4f}")
        print(f"pooled: mean_pos={sep['mean_proba_positive']:.4f} mean_neg={sep['mean_proba_negative']:.4f} "
              f"ratio={sep['pooled_ratio']:.1f}x cohens_d={sep['cohens_d']:.2f}")
        print(f"per-sample: median_ratio={sep['per_sample_ratio_median']:.1f}x "
              f"miss_rate={sep['miss_rate']:.1%} ({elapsed:.0f}s)")

        ckpt_path = args.out_dir / f"tornado_cnn_alpha_{alpha}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "loss_history": train_result["loss_history"],
            "val_loss_history": train_result["val_loss_history"],
            "training_hyperparameters": train_result["hyperparameters"],
            "model_hyperparameters": {"in_channels": len(train_ds.feature_names), "hidden_channels": 32},
            "feature_names": train_ds.feature_names,
        }, ckpt_path)

        results.append({
            "alpha": alpha,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "elapsed_sec": elapsed,
            "final_train_loss": train_result["loss_history"][-1],
            "final_val_loss": train_result["val_loss_history"][-1],
            "loss_history": train_result["loss_history"],
            "val_loss_history": train_result["val_loss_history"],
            "checkpoint": str(ckpt_path),
            **sep,
        })
        args.results.write_text(json.dumps(results, indent=2))

    print(f"\nSaved sweep results to {args.results}")


if __name__ == "__main__":
    main()
