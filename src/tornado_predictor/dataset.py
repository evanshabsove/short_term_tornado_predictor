"""
Combines feature extraction (features.py) and label assignment
(labels.py) across multiple HRRR runs/bins into one consolidated
training dataset.

Each (init_time, bin_index) is one sample. A bin spans 4 forecast hours
(time_bins.fxx_values_for_bin), but this module represents each bin with
a single representative fxx -- the bin's first hour (fxx=0 for bin 0,
fxx=4 for bin 1) -- rather than pulling and aggregating all 4 hourly
snapshots. This is a deliberate approximation to keep the Herbie network
cost proportional to the number of samples, not 4x that; aggregating the
full bin is a possible future refinement, not built here.

The saved dataset keeps the natural (sample, row, col) grid structure
with one variable per field/statistic plus a 0/1 label variable --
self-describing and consistent with how grid.py/features.py already
save their outputs. `to_training_arrays` is a separate, explicit
convenience function that flattens this into literal
(sample, grid, feature) + (sample, grid) numpy arrays for direct ML
consumption, since that flattening is lossy (discards which axis is
row vs. col) and shouldn't be baked into the storage format itself.

Labels are looked up from a precomputed positive-labels table (see
labels.build_positive_labels) rather than the SPC reports directly --
build that table once, pass it in here for every sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.features import extract_features
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.time_bins import fxx_values_for_bin


def representative_fxx(bin_index: int) -> int:
    """The single fxx used to stand in for an entire bin -- its first
    hour. See module docstring for why this is an approximation."""
    return next(iter(fxx_values_for_bin(bin_index)))


def label_grid_for_sample(labels_df: pd.DataFrame, init_time, bin_index: int, grid: HrrrCoarseGrid) -> np.ndarray:
    """A dense (n_rows, n_cols) 0/1 array for one (init_time, bin_index)
    sample, built by scattering the matching rows of the precomputed
    positive-labels table onto a zeros grid."""
    init_time = pd.Timestamp(init_time)
    subset = labels_df[(labels_df["init_time"] == init_time) & (labels_df["bin_index"] == bin_index)]
    label = np.zeros((grid.n_rows, grid.n_cols), dtype=np.int8)
    label[subset["row"].to_numpy(), subset["col"].to_numpy()] = 1
    return label


def build_sample(init_time, bin_index: int, grid: HrrrCoarseGrid, labels_df: pd.DataFrame) -> xr.Dataset:
    """Features + label for one (init_time, bin_index) sample, dims
    (row, col) only -- no sample dimension yet, no lat/lon (those are
    identical across every sample and are attached once in
    build_dataset instead of being duplicated per sample)."""
    fxx = representative_fxx(bin_index)
    features_ds = extract_features(init_time, fxx, grid).drop_vars(["lat", "lon"])
    label = label_grid_for_sample(labels_df, init_time, bin_index, grid)
    return features_ds.assign(label=(("row", "col"), label))


def build_dataset(run_bins: list[tuple], grid: HrrrCoarseGrid, labels_df: pd.DataFrame) -> xr.Dataset:
    """Builds the full consolidated dataset for a list of (init_time,
    bin_index) samples, concatenated along a new "sample" dimension."""
    samples = [build_sample(init_time, bin_index, grid, labels_df) for init_time, bin_index in run_bins]
    combined = xr.concat(samples, dim="sample")
    combined = combined.assign_coords(
        init_time=("sample", [pd.Timestamp(init_time) for init_time, _ in run_bins]),
        bin_index=("sample", [bin_index for _, bin_index in run_bins]),
        lat=(("row", "col"), grid.lat_center),
        lon=(("row", "col"), grid.lon_center),
    )
    # xr.concat keeps the first sample's attrs verbatim, which is
    # misleading here -- "init_time"/"fxx"/"valid_time" would describe
    # only the first of many samples. Those are already correctly
    # represented per-sample via the init_time/bin_index coords above;
    # keep only the attrs that are genuinely dataset-wide (title, the
    # Herbie search strings, which are the same for every sample).
    combined.attrs = {
        k: v for k, v in combined.attrs.items() if k == "title" or k.startswith("search_")
    }
    combined.attrs["description"] = (
        "Consolidated (sample, row, col) training dataset: pooled HRRR features "
        "and 0/1 tornado labels across multiple (init_time, bin_index) samples. "
        "Each sample's bin is represented by a single fxx snapshot (see "
        "representative_fxx) rather than an aggregate over the full bin."
    )
    return combined


def to_training_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Flattens ds into literal (sample, grid, feature) and (sample,
    grid) numpy arrays, where "grid" is row and col stacked together.
    Returns (X, y, feature_names)."""
    feature_vars = [v for v in ds.data_vars if v != "label"]
    stacked = ds.stack(grid=("row", "col"))
    X = np.stack([stacked[v].transpose("sample", "grid").values for v in feature_vars], axis=-1)
    y = stacked["label"].transpose("sample", "grid").values
    return X, y, feature_vars
