"""
Live (operational) feature extraction: finds the newest HRRR run whose
forecast files are all published, and builds the exact model input the
production ensemble was trained on -- a (N_BINS, n_features, 81, 138)
array, one plane per time bin.

This is a thin wrapper over the training-time path, not a second
implementation: features.extract_features pulls and pools each field,
dataset.representative_fxx picks the snapshot that stands in for each bin
(fxx=0 for bin 0, fxx=4 for bin 1), and the channels are stacked in the
checkpoint's own feature order. Training applies no normalization
(training.DenseGridDataset feeds raw pooled values, imputing NaN to 0), so
none is applied here either. The parity test in tests/test_live_features.py
checks this against a staged training file, so any drift between the live
and training paths is caught rather than silently changing predictions.

Timing (checked against the live AWS archive): a run's fxx=0 file lands
about init+50 min and fxx=4 about init+59 min, so a run is usable roughly
one hour after its init time. Bin 0 ([init, init+4h)) is therefore already
about an hour old when the forecast is issued; callers should present the
valid window, not "the next four hours".

Differences from DenseGridDataset, on purpose:
- Only the 14 base features are pulled (no updraft helicity, which the
  production ensemble does not use) -- 8 Herbie pulls per fxx, not 11.
- NaN in a base feature raises instead of being imputed to 0. The base
  fields have zero NaN across the whole training archive, so a NaN here
  means a bad pull, and silently feeding zeros would give a plausible but
  wrong forecast.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from herbie import Herbie

from tornado_predictor.dataset import representative_fxx
from tornado_predictor.features import extract_features, field_specs
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.time_bins import N_BINS, bin_valid_time_range

# fxx=0 is published ~50 min after init; starting the search there avoids
# probing runs that cannot be complete yet.
PUBLISH_LATENCY_MINUTES = 50
DEFAULT_MAX_LOOKBACK_HOURS = 6

LIVE_FIELD_SPECS: list[tuple[str, str]] = field_specs(include_uh=False)


@dataclass(frozen=True)
class LiveInputs:
    """X: (N_BINS, n_features, n_rows, n_cols) float32 in feature_names order.
    bins: per-bin dicts with bin_index, fxx, valid_start, valid_end (naive UTC)."""

    X: np.ndarray
    init_time: pd.Timestamp
    feature_names: list[str]
    bins: list[dict]
    pull_seconds: float


def _naive_utc(ts) -> pd.Timestamp:
    """Project convention: naive timestamps, treated as UTC (see time_bins.py)."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def run_is_complete(init_time) -> bool:
    """True if every fxx the bins use has its .idx published on AWS and the
    .idx lists every live search string. Reads only the small .idx files, no
    GRIB data. Any failure counts as "not complete yet" -- the caller retries
    later rather than treating an unpublished run as an error."""
    try:
        for fxx in sorted({representative_fxx(b) for b in range(N_BINS)}):
            H = Herbie(init_time, model="hrrr", product="sfc", fxx=fxx, priority="aws", verbose=False)
            if H.grib is None or H.idx is None:
                return False
            search_this = H.index_as_dataframe["search_this"]
            if not all(search_this.str.contains(s, regex=True).any() for _, s in LIVE_FIELD_SPECS):
                return False
        return True
    except Exception:
        return False


def latest_complete_run(
    now=None,
    max_lookback_hours: int = DEFAULT_MAX_LOOKBACK_HOURS,
    is_complete: Callable[[pd.Timestamp], bool] = run_is_complete,
) -> pd.Timestamp | None:
    """Newest hourly init time (naive UTC) whose files are all published, or
    None if none of the last max_lookback_hours+1 candidate runs is -- the
    caller should treat None as "retry later", not as an error.

    Candidates start at floor(now - PUBLISH_LATENCY_MINUTES) and step back
    one hour at a time. is_complete is injectable so tests need no network."""
    now = _naive_utc(pd.Timestamp.now(tz="UTC") if now is None else now)
    newest = (now - pd.Timedelta(minutes=PUBLISH_LATENCY_MINUTES)).floor("h")
    for hours_back in range(max_lookback_hours + 1):
        candidate = newest - pd.Timedelta(hours=hours_back)
        if is_complete(candidate):
            return candidate
    return None


def extract_live_inputs(
    init_time,
    grid: HrrrCoarseGrid,
    feature_names: list[str],
    *,
    cache_root: Path | None = None,
    cleanup: bool = True,
) -> LiveInputs:
    """Pulls and pools the live features for one run and stacks them in
    feature_names order (take it from the production bundle / a checkpoint,
    never hard-code it). Raises ValueError if a requested feature is not
    produced or any value is NaN.

    cleanup: delete this run's raw HRRR download cache afterward (same
    pattern as build_scaled.process_one_run), so a half-written cached file
    can never poison a later attempt. cache_root defaults to Herbie's
    configured save_dir."""
    init_time = _naive_utc(init_time)
    planes, bins = [], []
    t0 = time.time()
    try:
        for bin_index in range(N_BINS):
            fxx = representative_fxx(bin_index)
            ds = extract_features(init_time, fxx, grid, include_uh=False)
            missing = [n for n in feature_names if n not in ds.data_vars]
            if missing:
                raise ValueError(f"requested features {missing} are not produced by the live extraction path")
            plane = np.stack([ds[n].values for n in feature_names]).astype("float32")
            n_nan = int(np.isnan(plane).sum())
            if n_nan:
                raise ValueError(
                    f"{n_nan} NaN values in base features for init {init_time} fxx={fxx} -- refusing to "
                    "impute (training data has none, so this is a bad pull)"
                )
            planes.append(plane)
            lo, hi = bin_valid_time_range(init_time, bin_index)
            bins.append({"bin_index": bin_index, "fxx": fxx, "valid_start": lo, "valid_end": hi})
    finally:
        if cleanup:
            root = cache_root if cache_root is not None else _herbie_cache_root()
            shutil.rmtree(root / "hrrr" / f"{init_time:%Y%m%d}", ignore_errors=True)
    return LiveInputs(
        X=np.stack(planes),
        init_time=init_time,
        feature_names=list(feature_names),
        bins=bins,
        pull_seconds=time.time() - t0,
    )


def _herbie_cache_root() -> Path:
    """Herbie's configured save_dir (same lookup as build_scaled.hrrr_cache_root,
    kept local so this module does not import the dataset-build module)."""
    return Path(Herbie("2024-01-01 00:00", model="hrrr", product="sfc", fxx=0, verbose=False).save_dir)
