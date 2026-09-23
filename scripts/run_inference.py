"""
Load a trained checkpoint and run it on samples from a consolidated
dataset, producing per-cell tornado probability maps, a training-set-
fit sanity check against known positive labels, and a visualization.

This is NOT a generalization test -- see
src/tornado_predictor/inference.py's module docstring. The checkpoint's
pilot training data has only 6 samples; this only confirms the model
learned *something* associated with its own training labels, not that
it generalizes to unseen data.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --sample-index 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from tornado_predictor.inference import (
    assert_feature_order_matches,
    load_checkpoint,
    plot_sample,
    predict_proba,
    sanity_check_against_labels,
)
from tornado_predictor.training import DenseGridDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "tornado_cnn_pilot.pt"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"
DEFAULT_OUT_PATH = REPO_ROOT / "outputs" / "inference_pilot.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--sample-index", type=int, default=None, help="default: visualize every sample")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(str(args.checkpoint))
    dataset = DenseGridDataset(str(args.dataset))
    assert_feature_order_matches(dataset.feature_names, checkpoint)

    indices = [args.sample_index] if args.sample_index is not None else list(range(len(dataset)))

    ncols = min(3, len(indices))
    nrows = -(-len(indices) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False, constrained_layout=True
    )

    im = None
    for i, idx in enumerate(indices):
        X, y = dataset[idx]
        proba = predict_proba(model, X)
        stats = sanity_check_against_labels(proba, y)

        print(
            f"sample {idx}: {stats['n_positive_cells']} positive cells, "
            f"mean proba at positives={stats['mean_proba_positive']:.4f}, "
            f"mean proba at negatives={stats['mean_proba_negative']:.6f}, "
            f"max proba at negatives={stats['max_proba_negative']:.4f}"
        )
        if stats["n_positive_cells"] and stats["mean_proba_positive"] <= stats["mean_proba_negative"]:
            print("  WARNING: mean probability at positive cells is not higher than at negative cells")

        ax = axes[i // ncols][i % ncols]
        _, im = plot_sample(proba, y, title=f"sample {idx}", ax=ax)

    for j in range(len(indices), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="predicted probability", shrink=0.8)
    fig.suptitle("Predicted tornado probability vs. true positive cells (training-set fit, pilot dataset)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved visualization to {args.out}")


if __name__ == "__main__":
    main()
