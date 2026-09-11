"""
Smoke test for Herbie: pull surface-based CAPE from a specific past HRRR
run + forecast lead time, and confirm it loads into an xarray.DataArray
with sane values.

Usage:
    python scripts/smoke_test_herbie.py
"""

from herbie import Herbie

# A specific past HRRR run + lead time, chosen for a known active severe
# weather day so CAPE values are non-trivial (2024-05-06, PDS tornado
# outbreak in Oklahoma). 00Z cycle, 3h forecast lead time.
RUN_DATE = "2024-05-06 00:00"
FXX = 3
SEARCH_STRING = "CAPE:surface"


def main() -> None:
    print(f"Requesting HRRR run={RUN_DATE!r} fxx={FXX} ({SEARCH_STRING})")

    H = Herbie(
        RUN_DATE,
        model="hrrr",
        product="sfc",
        fxx=FXX,
    )

    print(f"GRIB source: {H.grib_source}")
    print(f"GRIB file:   {H.grib}")

    ds = H.xarray(SEARCH_STRING)

    print("\nDataset loaded:")
    print(ds)

    cape = ds["cape"]
    values = cape.values

    print("\n--- Sanity checks ---")
    print(f"Shape:        {cape.shape}")
    print(f"Valid time:   {ds.valid_time.values}")
    print(f"Min CAPE:     {values.min():.1f} J/kg")
    print(f"Max CAPE:     {values.max():.1f} J/kg")
    print(f"Mean CAPE:    {values.mean():.1f} J/kg")
    print(f"NaN fraction: {float((values != values).mean()):.4f}")

    assert values.shape[0] > 0 and values.shape[1] > 0, "Empty grid"
    assert not (values != values).all(), "CAPE field is entirely NaN"
    assert values.max() > 0, "Expected some non-zero CAPE on a severe weather day"

    print("\nSMOKE TEST PASSED: CAPE pulled and loaded into xarray successfully.")


if __name__ == "__main__":
    main()
