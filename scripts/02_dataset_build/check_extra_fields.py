"""
Live check of the extra candidate fields (features.EXTRA_FIELD_SPECS): pulls each
one for real across HRRR eras and prints shape, NaN count, min/mean/max, so a
wrong search string, a changed field meaning, or an empty field is seen BEFORE a
multi-day download. Reports; does not decide.

Eras: Sept 2014 (v1), mid 2016 (v2), just before / after the 2018-07-12 v3
upgrade, just before / after the 2020-12-02 v4 upgrade, and 2024.

Usage:
    python scripts/02_dataset_build/check_extra_fields.py [--fxx 0 5]
"""

from __future__ import annotations

import argparse

import numpy as np

from tornado_predictor.features import EXTRA_FIELD_SPECS, pull_hrrr_field

DATES = [
    "2014-09-02 12:00", "2016-06-01 12:00", "2018-06-01 12:00", "2018-08-01 12:00",
    "2020-11-01 12:00", "2021-01-15 12:00", "2024-05-01 12:00",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fxx", type=int, nargs="+", default=[0, 5])
    args = parser.parse_args()

    problems = []
    for date in DATES:
        for fxx in args.fxx:
            print(f"\n== {date} fxx={fxx}")
            for name, search in EXTRA_FIELD_SPECS:
                try:
                    a = pull_hrrr_field(date, fxx, search)
                except Exception as e:  # report every failure, keep going
                    problems.append((date, fxx, name, f"{type(e).__name__}: {e}"))
                    print(f"  {name:14s} FAILED {type(e).__name__}: {str(e)[:80]}")
                    continue
                n_nan = int(np.isnan(a).sum())
                if n_nan or np.nanmax(a) == np.nanmin(a):
                    problems.append((date, fxx, name, f"{n_nan} NaN / constant"))
                print(f"  {name:14s} shape={a.shape} nan={n_nan} min={np.nanmin(a):.4g} mean={np.nanmean(a):.4g} max={np.nanmax(a):.4g}")
    print(f"\n{len(problems)} problem(s)")
    for p in problems:
        print("  ", *p)


if __name__ == "__main__":
    main()
