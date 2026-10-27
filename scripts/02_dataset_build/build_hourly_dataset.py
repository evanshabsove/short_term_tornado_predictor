"""
Stage all 8 hourly HRRR snapshots (fxx 0-7) per run, then derive whole-bin
staging (features built from all 4 hours of each bin).

Runs are exactly the runs already `done` in the v2 manifest, so the whole-bin
dataset covers the same samples (and the same splits) as the existing
single-snapshot dataset. fxx 0 and 4 are copied from each run's existing v2
staged file; only fxx 1,2,3,5,6,7 are downloaded (--no-reuse pulls all 8).

LONG-RUNNING and resumable, like build_scaled_dataset.py: progress lives in a
manifest, re-running skips finished runs, safe to stop at any point.

Usage:
    # pilot: 20 runs spread across 2014-2025, separate staging dir
    python scripts/02_dataset_build/build_hourly_dataset.py --pilot 20 --workers 2
    # full pull
    python scripts/02_dataset_build/build_hourly_dataset.py --workers 4
    python scripts/02_dataset_build/build_hourly_dataset.py --workers 4 --retry-failed
    # the extra candidate fields (docs/feature_candidates_research.md), own store, all 8 hours pulled
    python scripts/02_dataset_build/build_hourly_dataset.py --feature-set extra --workers 4
    # local step only: derive whole-bin staging from whatever hourly files exist
    python scripts/02_dataset_build/build_hourly_dataset.py --derive-only
    # ... with the extra channels appended after the base 14 (needs both stores)
    python scripts/02_dataset_build/build_hourly_dataset.py --derive-only --include-extra
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tornado_predictor.build_hourly import derive_wholebin_staging, run_hourly_pull
from tornado_predictor.build_scaled import hrrr_cache_root, load_manifest
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM = REPO_ROOT / "data" / "interim"
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"
DEFAULT_STAGED_DIR = INTERIM / "scaled_build_v2"


def pilot_runs(done_runs: list[pd.Timestamp], n: int) -> list[pd.Timestamp]:
    """n runs evenly spaced through the sorted done list (so they span the
    whole 2014-2025 archive and every season), deterministic."""
    idx = np.unique(np.linspace(0, len(done_runs) - 1, n).round().astype(int))
    return [done_runs[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED_DIR, help="existing v2 staging (source of runs and of fxx 0/4)")
    parser.add_argument("--staging-dir", type=Path, default=None, help="hourly store (default: scaled_build_hourly, or ..._pilot with --pilot)")
    parser.add_argument("--wholebin-dir", type=Path, default=None, help="derived whole-bin staging (default: scaled_build_wholebin, or ..._pilot)")
    parser.add_argument("--workers", type=int, default=1, help="parallel dates (never parallel within a date)")
    parser.add_argument("--pilot", type=int, default=None, help="only N runs, evenly spaced across the archive")
    parser.add_argument("--no-reuse", action="store_true", help="pull all 8 hours instead of reusing the staged fxx 0/4")
    parser.add_argument("--feature-set", choices=["base", "extra"], default="base", help="which fields to pull; each has its own store and manifest")
    parser.add_argument("--include-extra", action="store_true", help="when deriving, append the extra channels (reads the extra store)")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--derive-only", action="store_true", help="skip pulling; derive whole-bin staging from existing hourly files")
    parser.add_argument("--skip-derive", action="store_true", help="pull only")
    args = parser.parse_args()

    suffix = "_pilot" if args.pilot else ""
    store = "" if args.feature_set == "base" else "_extra"
    staging_dir = args.staging_dir or INTERIM / f"scaled_build_hourly{store}{suffix}"
    extra_dir = INTERIM / f"scaled_build_hourly_extra{suffix}"
    base_dir = INTERIM / f"scaled_build_hourly{suffix}"
    wholebin_dir = args.wholebin_dir or INTERIM / f"scaled_build_wholebin{'_extra' if args.include_extra else ''}{suffix}"
    manifest_path = staging_dir / "manifest.json"

    v2 = load_manifest(args.staged_dir / "manifest.json")
    done_runs = sorted(pd.Timestamp(k) for k, v in v2.items() if v["status"] == "done")
    runs = pilot_runs(done_runs, args.pilot) if args.pilot else done_runs
    print(f"{len(runs)} runs ({len(done_runs)} done in the v2 manifest)")

    if not args.derive_only:
        start = time.time()
        manifest = run_hourly_pull(
            runs,
            grid_path=args.grid,
            staged_dir=None if (args.no_reuse or args.feature_set == "extra") else args.staged_dir,
            out_dir=staging_dir,
            manifest_path=manifest_path,
            cache_root=hrrr_cache_root(),
            workers=args.workers,
            retry_failed=args.retry_failed,
            feature_set=args.feature_set,
        )
        elapsed = time.time() - start
        n_done = sum(1 for r in runs if manifest.get(r.isoformat(), {}).get("status") == "done")
        n_failed = sum(1 for r in runs if manifest.get(r.isoformat(), {}).get("status") == "failed")
        print(f"\nManifest (these runs): {n_done} done, {n_failed} failed; elapsed {elapsed / 3600:.2f} h")

    if not args.skip_derive and args.feature_set == "base":
        grid = HrrrCoarseGrid.from_netcdf(args.grid)
        labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])
        manifest = derive_wholebin_staging(
            args.staging_dir or base_dir, wholebin_dir, runs, labels_df, grid,
            extra_hourly_dir=extra_dir if args.include_extra else None,
        )
        print(f"Derived whole-bin staging for {len(manifest)} runs -> {wholebin_dir}")


if __name__ == "__main__":
    main()
