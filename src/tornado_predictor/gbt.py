"""
Data prep for a gradient-boosted tree model family -- a genuinely
different alternative to the CNN/U-Net line (model.py/unet.py),
motivated by literature finding a histogram gradient-boosted tree
(HGBT) beat a deep U-Net on a closely analogous CAM-grid severe-weather
task. See scripts/train_gbt.py for the actual model
(sklearn.ensemble.HistGradientBoostingClassifier).

Two small, pure/testable functions -- the model itself is a standard
sklearn estimator, no custom class needed.

Per-cell downsampling is the key idea this module exists for:
CLAUDE.md's "Class imbalance strategy" section is explicit that
per-cell downsampling "isn't really executable for a spatial CNN (can't
drop arbitrary pixels from a grid map without breaking the
convolutional receptive field)" -- which is exactly why
training.QuietMapDownsampler operates at the map level instead. A
tabular model has no such constraint: each grid cell is an independent
row, so per-cell downsampling is not just possible here but the
standard way to handle extreme imbalance for tree-based models.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tornado_predictor.dataset import to_training_arrays


def flatten_for_gbt(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """dataset.to_training_arrays(ds) gives (sample, grid, feature) /
    (sample, grid) arrays -- reshapes one step further to one row per
    grid cell: (sample*grid, feature) / (sample*grid,)."""
    X, y, feature_names = to_training_arrays(ds)
    n_features = X.shape[-1]
    return X.reshape(-1, n_features), y.reshape(-1), feature_names


def downsample_negative_cells(
    X: np.ndarray, y: np.ndarray, negative_ratio: float = 20.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Keeps every positive (y==1) row; randomly samples
    negative_ratio * n_positive negative (y==0) rows without
    replacement. Raises if fewer negatives are available than
    requested -- a loud failure beats a silent under-sample."""
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)

    n_negative_wanted = int(round(negative_ratio * len(positive_idx)))
    if n_negative_wanted > len(negative_idx):
        raise ValueError(
            f"requested {n_negative_wanted} negative rows (ratio={negative_ratio} x "
            f"{len(positive_idx)} positives) but only {len(negative_idx)} negative rows exist"
        )

    rng = np.random.default_rng(seed)
    sampled_negative_idx = rng.choice(negative_idx, size=n_negative_wanted, replace=False)

    keep_idx = np.concatenate([positive_idx, sampled_negative_idx])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]
