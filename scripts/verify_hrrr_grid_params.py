"""
One-time live check: pull a real HRRR grib2 file via Herbie and confirm
its GRIB_* grid-definition keys match the hardcoded HRRR_* constants in
src/tornado_predictor/grid.py. Requires network access -- do not run this
as part of normal grid building (see scripts/build_grid.py for that).

Usage:
    python scripts/verify_hrrr_grid_params.py
"""

from __future__ import annotations

import pygrib
from herbie import Herbie

from tornado_predictor import grid as g

RUN_DATE = "2024-05-06 00:00"
FXX = 3
SEARCH_STRING = "CAPE:surface"


def main() -> None:
    H = Herbie(RUN_DATE, model="hrrr", product="sfc", fxx=FXX)
    local_path = H.download(SEARCH_STRING)

    grbs = pygrib.open(str(local_path))
    grb = grbs.message(1)

    print(f"pygrib projparams: {grb.projparams}\n")

    shape_of_earth = grb["shapeOfTheEarth"]
    print(f"  {'OK' if shape_of_earth == 6 else 'MISMATCH'}: shapeOfTheEarth = {shape_of_earth} (expected 6, spherical with explicit radius)\n")

    # LoV/lon_0 keys come back in GRIB's 0-360 convention; HRRR_LON_0_DEG is
    # stored in proj4's -180..180 convention, so normalize mod 360 to compare.
    checks = [
        ("Latin1InDegrees", grb["Latin1InDegrees"], g.HRRR_LAT_1_DEG),
        ("Latin2InDegrees", grb["Latin2InDegrees"], g.HRRR_LAT_1_DEG),
        ("LaDInDegrees", grb["LaDInDegrees"], g.HRRR_LAT_0_DEG),
        ("LoVInDegrees", grb["LoVInDegrees"] % 360, g.HRRR_LON_0_DEG % 360),
        ("latitudeOfFirstGridPointInDegrees", grb["latitudeOfFirstGridPointInDegrees"], g.HRRR_LA1_DEG),
        ("longitudeOfFirstGridPointInDegrees", grb["longitudeOfFirstGridPointInDegrees"], g.HRRR_LO1_DEG),
        ("Nx", grb["Nx"], g.HRRR_NX),
        ("Ny", grb["Ny"], g.HRRR_NY),
        ("DxInMetres", grb["DxInMetres"], g.HRRR_DX_M),
        ("DyInMetres", grb["DyInMetres"], g.HRRR_DY_M),
        ("radius", grb["radius"], g.HRRR_EARTH_RADIUS_M),
    ]

    all_ok = shape_of_earth == 6
    for name, actual, expected in checks:
        ok = abs(float(actual) - float(expected)) < 1e-3
        all_ok &= ok
        print(f"  {'OK' if ok else 'MISMATCH'}: {name} = {actual} (expected {expected})")

    if not all_ok:
        raise SystemExit("\nHRRR grid parameter mismatch -- update grid.py constants before trusting the module.")
    print("\nAll HRRR native grid parameters verified against live grib file.")


if __name__ == "__main__":
    main()
