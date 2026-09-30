"""
Adds new HRRR fields (default: the 3 updraft helicity fields, see
features.py's module docstring) to every already-successfully-staged
run from the v2 build, without re-pulling the fields that were already
correctly extracted. Writes augmented copies to a NEW staging directory
(default data/interim/scaled_build_v3/), never touching the original v2
staged files, then rebuilds the final train/val datasets from the
augmented copies via build_scaled.finalize() (unchanged, fully generic
to which fields exist).

Resumable, same spirit as build_scaled_dataset.py: progress is tracked
in its own manifest (separate from v2's, since the unit of work here is
"has this run been augmented yet", a different question from v2's "was
this run successfully pulled").

Usage:
    python scripts/augment_dataset_fields.py
    python scripts/augment_dataset_fields.py --retry-failed
    python scripts/augment_dataset_fields.py --finalize-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.augment_features import augment_one_run
from tornado_predictor.build_scaled import finalize, hrrr_cache_root as get_hrrr_cache_root, load_manifest, save_manifest
from tornado_predictor.features import FIELD_SPECS, NEW_FIELD_NAMES_2026_09
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_SRC_MANIFEST = REPO_ROOT / "data" / "interim" / "scaled_build_v2" / "manifest.json"
DEFAULT_SRC_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
DEFAULT_DST_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v3"
DEFAULT_DST_MANIFEST = DEFAULT_DST_STAGING_DIR / "manifest.json"
DEFAULT_OUT_TRAIN = REPO_ROOT / "data" / "processed" / "training_dataset_train_full_v3.nc"
DEFAULT_OUT_VAL = REPO_ROOT / "data" / "processed" / "training_dataset_val_full_v3.nc"

DEFAULT_NEW_FIELDS = NEW_FIELD_NAMES_2026_09


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--src-manifest", type=Path, default=DEFAULT_SRC_MANIFEST)
    parser.add_argument("--src-staging-dir", type=Path, default=DEFAULT_SRC_STAGING_DIR)
    parser.add_argument("--dst-staging-dir", type=Path, default=DEFAULT_DST_STAGING_DIR)
    parser.add_argument("--dst-manifest", type=Path, default=DEFAULT_DST_MANIFEST)
    parser.add_argument("--out-train", type=Path, default=DEFAULT_OUT_TRAIN)
    parser.add_argument("--out-val", type=Path, default=DEFAULT_OUT_VAL)
    parser.add_argument("--fields", type=str, nargs="+", default=DEFAULT_NEW_FIELDS, help="field names (must be in features.FIELD_SPECS) to add")
    parser.add_argument("--finalize-only", action="store_true", help="skip augmenting, just rebuild train/val from already-augmented runs")
    parser.add_argument("--retry-failed", action="store_true", help="also retry runs previously marked failed")
    args = parser.parse_args()

    new_field_specs = [(n, s) for n, s in FIELD_SPECS if n in args.fields]
    assert len(new_field_specs) == len(args.fields), f"not all of {args.fields} found in FIELD_SPECS"
    print(f"augmenting with: {new_field_specs}")

    src_manifest = load_manifest(args.src_manifest)
    src_done = {k: v["message"] for k, v in src_manifest.items() if v["status"] == "done"}
    print(f"{len(src_done)} runs completed in the v2 build")

    dst_manifest = load_manifest(args.dst_manifest)

    if not args.finalize_only:
        grid = HrrrCoarseGrid.from_netcdf(args.grid)
        cache_root = get_hrrr_cache_root()

        todo = [
            key for key in src_done
            if key not in dst_manifest
            or (dst_manifest[key]["status"] == "failed" and args.retry_failed)
        ]
        print(f"{len(todo)} runs to augment ({len(src_done) - len(todo)} already done)")

        for i, key in enumerate(todo):
            src_path = Path(src_done[key])
            dst_path = args.dst_staging_dir / src_path.name
            success, message = augment_one_run(src_path, dst_path, new_field_specs, grid, cache_root)
            dst_manifest[key] = {"status": "done" if success else "failed", "message": message}
            save_manifest(args.dst_manifest, dst_manifest)
            status = "OK" if success else "FAILED"
            print(f"[{i + 1}/{len(todo)}] {key}: {status}" + ("" if success else f" -- {message}"))

    n_done = sum(1 for v in dst_manifest.values() if v["status"] == "done")
    n_failed = sum(1 for v in dst_manifest.values() if v["status"] == "failed")
    print(f"\nManifest: {n_done} done, {n_failed} failed")

    completed_runs = sorted(pd.Timestamp(k) for k, v in dst_manifest.items() if v["status"] == "done")
    result = finalize(args.dst_staging_dir, completed_runs, args.out_train, args.out_val)
    print(f"Finalized: {result}")
    print(f"Saved train -> {args.out_train}")
    print(f"Saved val   -> {args.out_val}")


if __name__ == "__main__":
    main()
