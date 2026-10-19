"""
Read-only smoke test of the live feature path against the real, current
HRRR archive: finds the latest complete run, extracts the model inputs, and
compares each channel's range with a sample of the training data, so a unit
change or an HRRR upgrade that shifts a field's distribution gets noticed.

Prints timing, then per channel: live min / max next to the training sample's
min / p99.9 / max. A live value outside the training min/max is flagged
"OUTSIDE" -- informational, not a failure: a genuinely extreme event can
exceed a sample's range. A whole channel shifted (e.g. t2m now in degrees C,
or everything zero) is the case this exists to catch.

Usage:
    python scripts/09_serving/live_features_smoke.py
    python scripts/09_serving/live_features_smoke.py --init-time "2026-10-18 23:00"   # skip discovery
    python scripts/09_serving/live_features_smoke.py --bundle models/production/manifest.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.ensemble import load_bundle, predict_ensemble
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.inference import load_checkpoint
from tornado_predictor.live_features import extract_live_inputs, latest_complete_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_CHECKPOINT = REPO_ROOT / "models" / "ema_study" / "seed10" / "tornado_unet_heldout_2020_ema0.9999.pt"
DEFAULT_TRAINING_SAMPLE = REPO_ROOT / "data" / "processed" / "training_dataset_val_full_v4.nc"
N_REFERENCE_SAMPLES = 300


def reference_ranges(path: Path, feature_names: list[str], n: int) -> dict[str, tuple[float, float, float]]:
    """(min, p99.9, max) per feature over an evenly spaced sample of n training samples."""
    with xr.open_dataset(path) as ds:
        idx = np.linspace(0, ds.sizes["sample"] - 1, n).astype(int)
        sub = ds.isel(sample=idx)
        return {
            name: (
                float(sub[name].min()),
                float(np.nanquantile(sub[name].values, 0.999)),
                float(sub[name].max()),
            )
            for name in feature_names
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-time", default=None, help="skip discovery and use this init time (UTC)")
    parser.add_argument("--bundle", type=Path, default=None, help="also run this production bundle on the live inputs")
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--training-sample", type=Path, default=DEFAULT_TRAINING_SAMPLE)
    args = parser.parse_args()

    _, checkpoint = load_checkpoint(str(args.reference_checkpoint))
    names = checkpoint["feature_names"]
    grid = build_coarse_grid()

    t0 = time.time()
    init_time = pd.Timestamp(args.init_time) if args.init_time else latest_complete_run()
    if init_time is None:
        print("no complete HRRR run found in the lookback window -- the scheduler would retry later")
        return
    print(f"latest complete run: {init_time} UTC (found in {time.time() - t0:.1f}s; "
          f"now {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC)")

    live = extract_live_inputs(init_time, grid, names)
    print(f"extracted {live.X.shape} float32 in {live.pull_seconds:.0f}s")
    for b in live.bins:
        print(f"  bin {b['bin_index']}: fxx={b['fxx']}  valid {b['valid_start']:%d %H:%MZ} -> {b['valid_end']:%d %H:%MZ}")

    ref = reference_ranges(args.training_sample, names, N_REFERENCE_SAMPLES) if args.training_sample.exists() else None
    print(f"\n{'feature':18s} {'bin':>3s} {'live min':>11s} {'live max':>11s}   training sample min / p99.9 / max")
    for c, name in enumerate(names):
        for b in range(live.X.shape[0]):
            lo, hi = float(live.X[b, c].min()), float(live.X[b, c].max())
            note = ""
            if ref:
                rmin, rp, rmax = ref[name]
                note = f"{rmin:11.2f} {rp:11.2f} {rmax:11.2f}"
                if lo < rmin or hi > rmax:
                    note += "   OUTSIDE"
            print(f"{name:18s} {b:3d} {lo:11.2f} {hi:11.2f}   {note}")

    if args.bundle:
        bundle = load_bundle(args.bundle)
        pred = predict_ensemble(bundle, live.X, feature_names=names)
        print(f"\nbundle {bundle.bundle_id}: mean prob per bin {pred.mean.mean(axis=(1, 2))}, "
              f"max {pred.mean.max(axis=(1, 2))}, max spread {pred.spread.max(axis=(1, 2))}")


if __name__ == "__main__":
    main()
