"""
Run-level sequence dataset for the ConvLSTM study (docs/convlstm_spec.md, ticket T1).

Every earlier model sees one `(channel, row, col)` map per (run, bin) sample. A sequence
model sees one whole RUN: the 8 hourly snapshots (fxx 0-7) of the 14 base features as an
`(8, 14, 81, 138)` input, and both bins' labels as a `(2, 81, 138)` target.

Inputs are read lazily, one run per `__getitem__`, from the hourly store
(`build_hourly.py`, `(fxx, row, col)` per field; ~5 ms per run, the full store is 23 GB so
it is never held in memory). Labels come from the whole-bin staging
(`derive_wholebin_run`), whose `label` variable is identical to the v4 labels on every
shared sample, so a sequence model is trained against exactly the labels arm A used.

Sample order is the contract that lets the existing evaluation code score this model
against arm A's cached scores: `flatten_run_bins` turns `(run, bin, ...)` back into A's
`(run, bin)` sample order, run-major with the bin index varying fastest, as
`combine_staged_runs` produces it. That only holds if `runs` is passed in the same order as
A's run_bins (manifest order), which `runs_from_run_bins` preserves.

Deliberately NOT done here: any scaling (the base 14 stay raw, like every model so far) and
NaN imputation (the base fields have none; a NaN means a bad file and raises instead of
being silently zeroed, unlike `DenseGridDataset`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from tornado_predictor.build_hourly import BASE_FEATURE_NAMES, HOURLY_FXX
from tornado_predictor.build_scaled import run_staging_path
from tornado_predictor.time_bins import N_BINS


def runs_from_run_bins(run_bins: list[tuple]) -> list[pd.Timestamp]:
    """Unique runs of a `(init_time, bin_index)` list in first-appearance order -- the
    order a fold's runs must keep for `flatten_run_bins` to match arm A's sample order."""
    seen: dict[pd.Timestamp, None] = {}
    for init_time, _ in run_bins:
        seen.setdefault(pd.Timestamp(init_time), None)
    return list(seen)


def flatten_run_bins(arr: np.ndarray) -> np.ndarray:
    """`(n_runs, N_BINS, ...)` -> `(n_runs * N_BINS, ...)` in arm A's `(run, bin)` order."""
    if arr.ndim < 2 or arr.shape[1] != N_BINS:
        raise ValueError(f"expected shape (n_runs, {N_BINS}, ...), got {arr.shape}")
    return arr.reshape(arr.shape[0] * N_BINS, *arr.shape[2:])


class DenseSequenceDataset(Dataset):
    """One run per item: `(X, y)` with X `(8, n_features, row, col)` float32 and y
    `(N_BINS, row, col)` float32.

    `is_active` (per run) is True if either bin has a positive cell -- the run-level
    notion of "active" the quiet downsampler needs (a run is quiet only if BOTH bins are
    all-negative). `is_active_by_bin` `(n_runs, N_BINS)` is kept for reporting.
    Labels are read once at construction (small); features only on access.
    """

    def __init__(
        self,
        hourly_dir: Path,
        label_dir: Path,
        runs: list,
        feature_names: list[str] | None = None,
    ):
        self.hourly_dir = Path(hourly_dir)
        self.label_dir = Path(label_dir)
        self.runs = [pd.Timestamp(r) for r in runs]
        if len(set(self.runs)) != len(self.runs):
            raise ValueError("duplicate runs")
        self.feature_names = list(BASE_FEATURE_NAMES if feature_names is None else feature_names)
        self.hourly_paths = [run_staging_path(self.hourly_dir, r) for r in self.runs]
        missing = [str(p) for p in self.hourly_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} hourly run file(s) missing, first: {missing[0]}")

        labels = []
        for r in self.runs:
            with xr.open_dataset(run_staging_path(self.label_dir, r)) as ds:
                y = ds["label"].values.astype("float32")
            if y.shape[0] != N_BINS:
                raise ValueError(f"{r}: label has {y.shape[0]} bins, expected {N_BINS}")
            labels.append(y)
        self.y = np.stack(labels) if labels else np.zeros((0, N_BINS, 0, 0), dtype="float32")
        self.is_active_by_bin = self.y.sum(axis=(2, 3)) > 0
        self.is_active = self.is_active_by_bin.any(axis=1)

    def __len__(self) -> int:
        return len(self.runs)

    def load_inputs(self, idx: int) -> np.ndarray:
        with xr.open_dataset(self.hourly_paths[idx]) as ds:
            if list(ds["fxx"].values) != list(HOURLY_FXX):
                raise ValueError(f"{self.runs[idx]}: fxx axis is {list(ds['fxx'].values)}, expected {list(HOURLY_FXX)}")
            # (channel, fxx, row, col) -> (fxx, channel, row, col): time first, as a sequence model expects
            x = np.stack([ds[name].values for name in self.feature_names]).transpose(1, 0, 2, 3)
        x = x.astype("float32", copy=False)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.runs[idx]}: non-finite values in the hourly inputs")
        return x

    def __getitem__(self, idx: int):
        return self.load_inputs(idx), self.y[idx]
