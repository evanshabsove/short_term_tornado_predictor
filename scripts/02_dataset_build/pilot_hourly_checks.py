"""
Pilot gates for the hourly/whole-bin build (run after build_hourly_dataset.py
--pilot N). Reports; does not decide.

  (i)   every needed .idx field is present at fxx 1,2,3,5,6,7 for each run
  (ii)  reuse check: fresh pulls of fxx 0 and 4 equal the staged copies
  (iv)  sanity of the hourly files: NaNs, hour-to-hour smoothness, and whether
        the whole-bin features stay inside the range of the existing v2 data

(iii), throughput, is the "elapsed" line build_hourly_dataset.py prints.

Usage:
    python scripts/02_dataset_build/pilot_hourly_checks.py [--n-reuse 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.build_hourly import (
    BASE_FEATURE_NAMES,
    STAGED_FXX_TO_SAMPLE,
    aggregate_bin,
    hourly_run_path,
    pull_snapshot,
    staged_snapshot,
)
from tornado_predictor.build_scaled import load_manifest, run_staging_path
from tornado_predictor.features import field_available, field_specs
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM = REPO_ROOT / "data" / "interim"
NEW_FXX = (1, 2, 3, 5, 6, 7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc")
    parser.add_argument("--hourly-dir", type=Path, default=INTERIM / "scaled_build_hourly_pilot")
    parser.add_argument("--staged-dir", type=Path, default=INTERIM / "scaled_build_v2")
    parser.add_argument("--n-reuse", type=int, default=4, help="how many pilot runs to re-pull fxx 0/4 for (slow)")
    args = parser.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    pilot = load_manifest(args.hourly_dir / "manifest.json")
    runs = sorted(pd.Timestamp(k) for k, v in pilot.items() if v["status"] == "done")
    print(f"{len(runs)} staged pilot runs")

    print("\n(i) .idx field availability at the new hours")
    missing = []
    for run in runs:
        for fxx in NEW_FXX:
            for name, search in field_specs(include_uh=False):
                if not field_available(run, fxx, search):
                    missing.append((run, fxx, name))
    print(f"  missing: {len(missing)}", *missing[:10], sep="\n    " if missing else "")

    print("\n(ii) reuse check (fresh pull vs staged copy)")
    step = max(1, len(runs) // max(1, args.n_reuse))
    worst = 0.0
    for run in runs[::step][: args.n_reuse]:
        for fxx, sample in STAGED_FXX_TO_SAMPLE.items():
            fresh = pull_snapshot(run, fxx, grid)
            old = staged_snapshot(run_staging_path(args.staged_dir, run), sample)
            diff = max(float(np.max(np.abs(fresh[n] - old[n]))) for n in BASE_FEATURE_NAMES)
            worst = max(worst, diff)
            print(f"  {run} fxx={fxx}: max abs diff {diff:.3g}")
    print(f"  worst: {worst:.3g}")

    print("\n(iv) sanity")
    v2_ranges = {n: [np.inf, -np.inf] for n in BASE_FEATURE_NAMES}
    for run in runs:
        with xr.open_dataset(run_staging_path(args.staged_dir, run)) as old:
            for n in BASE_FEATURE_NAMES:
                v2_ranges[n][0] = min(v2_ranges[n][0], float(old[n].min()))
                v2_ranges[n][1] = max(v2_ranges[n][1], float(old[n].max()))
    n_nan = 0
    outside = {}
    for run in runs:
        with xr.open_dataset(hourly_run_path(args.hourly_dir, run)) as h:
            h = h.load()
        for n in BASE_FEATURE_NAMES:
            n_nan += int(np.isnan(h[n].values).sum())
            for b in (0, 1):
                agg = aggregate_bin(h, b)[n].values
                lo, hi = v2_ranges[n]
                if agg.min() < lo - 1e-6 or agg.max() > hi + 1e-6:
                    outside[n] = outside.get(n, 0) + 1
    print(f"  NaNs in hourly files: {n_nan}")
    print(f"  (run,bin) aggregates outside the pilot runs' v2 range: {outside or 'none'}")
    print("  largest hour-to-hour change of each channel's domain mean, as % of its range, over all pilot runs:")
    for n in BASE_FEATURE_NAMES:
        worst_jump = 0.0
        for run in runs:
            with xr.open_dataset(hourly_run_path(args.hourly_dir, run)) as h:
                series = h[n].mean(("row", "col")).values
            span = v2_ranges[n][1] - v2_ranges[n][0]
            worst_jump = max(worst_jump, float(np.max(np.abs(np.diff(series)))) / span * 100 if span else 0.0)
        print(f"    {n}: {worst_jump:.2f}%")


if __name__ == "__main__":
    main()
