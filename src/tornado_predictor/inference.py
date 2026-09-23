"""
Loads a trained checkpoint (scripts/train_model.py) and runs it on
samples from a consolidated dataset (training.DenseGridDataset),
producing a per-cell tornado probability map plus a training-set-fit
sanity check against the known positive labels for that sample.

This is NOT a generalization test. The checkpoint's pilot training
data has only 6 samples, so a high probability at known-positive cells
only confirms the model learned to associate its own training labels
with something in the input (memorization) -- not that it would
perform well on unseen data. See training.py / dataset.py for why the
pilot is intentionally scoped this small.
"""

from __future__ import annotations

import numpy as np
import torch

from tornado_predictor.model import TornadoCNN


def load_checkpoint(path: str) -> tuple[TornadoCNN, dict]:
    """Reconstructs the exact model architecture from the checkpoint's
    saved hyperparameters, loads its weights, and sets eval mode."""
    checkpoint = torch.load(path, weights_only=False)
    model = TornadoCNN(**checkpoint["model_hyperparameters"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def assert_feature_order_matches(dataset_feature_names: list[str], checkpoint: dict) -> None:
    """The dataset's feature channel order must exactly match the
    order the checkpoint was trained on -- a silent mismatch here
    would produce plausible-looking but meaningless predictions (wrong
    channel fed into the position the model learned to associate with
    e.g. CAPE)."""
    if dataset_feature_names != checkpoint["feature_names"]:
        raise ValueError(
            f"dataset feature order {dataset_feature_names} does not match "
            f"the checkpoint's training-time feature order {checkpoint['feature_names']} "
            "-- predictions would be silently wrong if channels don't line up"
        )


def predict_proba(model: TornadoCNN, X: np.ndarray) -> np.ndarray:
    """X: (channels, row, col) -> probability map (row, col), via
    sigmoid on the model's raw logits (see model.py)."""
    with torch.no_grad():
        x = torch.from_numpy(X).unsqueeze(0)
        logits = model(x)
        return torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()


def sanity_check_against_labels(proba: np.ndarray, label: np.ndarray) -> dict:
    """Training-set-fit check: does the model assign higher probability
    to cells it was told are positive than to the rest? Not a
    generalization test -- see module docstring."""
    positive = label > 0
    return {
        "n_positive_cells": int(positive.sum()),
        "mean_proba_positive": float(proba[positive].mean()) if positive.any() else float("nan"),
        "min_proba_positive": float(proba[positive].min()) if positive.any() else float("nan"),
        "mean_proba_negative": float(proba[~positive].mean()),
        "max_proba_negative": float(proba[~positive].max()),
    }


def plot_sample(proba: np.ndarray, label: np.ndarray, title: str, ax=None):
    """Probability heatmap (viridis -- perceptually uniform, colorblind-
    safe sequential colormap for a magnitude field) with true positive
    cells marked distinctly on top (a red X reads clearly against
    viridis at any probability level, unlike a same-hue marker that
    could blend into a bright cell). origin='lower' since grid row 0 is
    the domain's southwest corner (grid.py), so the plot reads
    north-up, matching the real geography."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(proba, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    pos_rows, pos_cols = np.where(label > 0)
    if pos_rows.size:
        ax.scatter(pos_cols, pos_rows, marker="x", s=60, c="red", linewidths=2, label="true positive cell")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    return ax, im
