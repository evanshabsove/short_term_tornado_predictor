"""
Pull HRRR severe-weather fields for one (init time, forecast hour) via
Herbie and pool each onto the ~40km grid (see
src/tornado_predictor/features.py and src/tornado_predictor/grid.py).

Usage:
    python scripts/extract_features.py --init-time "2024-05-06 00:00" --fxx 3
    python scripts/extract_features.py --init-time "2024-05-06 00:00" --fxx 3 --grid data/processed/hrrr_coarse_grid_stride13.nc --out data/interim/features_2024050600_f03.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tornado_predictor.features import extract_features
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
INTERIM_DIR = REPO_ROOT / "data" / "interim"


def print_sanity_summary(ds) -> None:
    field_names = sorted({v.rsplit("_", 1)[0] for v in ds.data_vars if v not in ("lat", "lon")})
    for name in field_names:
        mean = ds[f"{name}_mean"].values
        max_ = ds[f"{name}_max"].values
        ok = bool((mean <= max_ + 1e-6).all())
        print(
            f"  {name:14s} mean[min/mean/max]={mean.min():9.2f}/{mean.mean():9.2f}/{mean.max():9.2f}"
            f"  max[min/mean/max]={max_.min():9.2f}/{max_.mean():9.2f}/{max_.max():9.2f}"
            f"  {'OK' if ok else 'MEAN>MAX VIOLATION'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-time", required=True, help='e.g. "2024-05-06 00:00"')
    parser.add_argument("--fxx", type=int, required=True)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    ds = extract_features(args.init_time, args.fxx, grid)

    init_compact = args.init_time.replace("-", "").replace(":", "").replace(" ", "")
    out_path = args.out or INTERIM_DIR / f"features_{init_compact}_f{args.fxx:02d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    print(f"Extracted features for init_time={args.init_time} fxx={args.fxx}")
    print_sanity_summary(ds)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
