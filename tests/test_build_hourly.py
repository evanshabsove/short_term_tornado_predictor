import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tornado_predictor.build_hourly import (
    BASE_FEATURE_NAMES,
    EXTRA_FEATURE_NAMES,
    FEATURE_SETS,
    HOURLY_FXX,
    STAGED_FXX_TO_SAMPLE,
    aggregate_bin,
    build_hourly_run,
    derive_wholebin_run,
    derive_wholebin_staging,
    group_runs_by_date,
    hourly_run_path,
    process_one_hourly_run,
    run_hourly_pull,
)
from tornado_predictor.build_scaled import combine_staged_runs, load_manifest, run_staging_path
from tornado_predictor.features import field_specs
from tornado_predictor.grid import build_coarse_grid

GRID = build_coarse_grid()
SHAPE = (GRID.n_rows, GRID.n_cols)
INIT = pd.Timestamp("2021-12-10 21:00:00")


def _fake_pull_factory(calls=None):
    """Every feature is the constant `fxx` (so a bin's expected aggregate is
    easy to state by hand); records each (init, fxx) it is asked for."""

    def pull(init_time, fxx, grid):
        if calls is not None:
            calls.append(fxx)
        return {name: np.full(SHAPE, float(fxx), dtype="float32") for name in BASE_FEATURE_NAMES}

    return pull


def _write_staged_v2_run(staging_dir, init_time, values=(-1.0, -2.0)):
    """A v2-format staged file whose bin-0 / bin-1 samples are constants, so
    reuse of fxx 0 / fxx 4 is unmistakable."""
    data = {
        name: (("sample", "row", "col"), np.stack([np.full(SHAPE, v, dtype="float32") for v in values]))
        for name in BASE_FEATURE_NAMES
    }
    data["label"] = (("sample", "row", "col"), np.zeros((2, *SHAPE), dtype=np.int8))
    path = run_staging_path(staging_dir, init_time)
    staging_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(data).to_netcdf(path)


def _labels(rows):
    df = pd.DataFrame(rows, columns=["event_id", "init_time", "bin_index", "row", "col"])
    return df.astype({"bin_index": "int64", "row": "int64", "col": "int64", "init_time": "datetime64[ns]"})


def test_base_feature_names_match_feature_extraction_order():
    # extract_features' order: FIELD_SPECS order with the two shear components
    # collapsed into one magnitude field appended last.
    physical = [n for n, _ in field_specs(include_uh=False) if n not in ("shear_u_0_6km", "shear_v_0_6km")]
    expected = [f"{n}_{s}" for n in physical + ["shear_0_6km"] for s in ("mean", "max")]
    assert BASE_FEATURE_NAMES == expected


def test_staged_fxx_are_the_old_representative_hours():
    assert STAGED_FXX_TO_SAMPLE == {0: 0, 4: 1}


def test_build_hourly_run_reuses_staged_fxx_0_and_4_and_pulls_the_other_six(tmp_path):
    _write_staged_v2_run(tmp_path, INIT)
    calls = []
    ds = build_hourly_run(INIT, GRID, tmp_path, pull=_fake_pull_factory(calls))

    assert sorted(calls) == [1, 2, 3, 5, 6, 7]
    assert list(ds["fxx"].values) == list(HOURLY_FXX)
    assert ds["cape_mean"].dims == ("fxx", "row", "col")
    assert float(ds["cape_mean"].sel(fxx=0).mean()) == -1.0  # from staged sample 0
    assert float(ds["cape_mean"].sel(fxx=4).mean()) == -2.0  # from staged sample 1
    assert float(ds["cape_mean"].sel(fxx=5).mean()) == 5.0  # pulled
    assert ds.attrs["reused_fxx"] == "0,4"
    assert list(ds.data_vars) == BASE_FEATURE_NAMES


def test_build_hourly_run_without_staged_dir_pulls_all_eight():
    calls = []
    ds = build_hourly_run(INIT, GRID, None, pull=_fake_pull_factory(calls))
    assert sorted(calls) == list(HOURLY_FXX)
    assert ds.attrs["reused_fxx"] == ""


def test_aggregate_bin_means_the_mean_channel_and_maxes_the_max_channel():
    ds = build_hourly_run(INIT, GRID, None, pull=_fake_pull_factory())

    bin0 = aggregate_bin(ds, 0)  # hours 0,1,2,3
    bin1 = aggregate_bin(ds, 1)  # hours 4,5,6,7

    assert list(bin0.data_vars) == BASE_FEATURE_NAMES
    assert float(bin0["cape_mean"].mean()) == pytest.approx(1.5)
    assert float(bin0["cape_max"].mean()) == pytest.approx(3.0)
    assert float(bin1["t2m_mean"].mean()) == pytest.approx(5.5)
    assert float(bin1["t2m_max"].mean()) == pytest.approx(7.0)
    assert bin0["cape_mean"].dtype == np.float32


def test_aggregate_bin_max_catches_a_single_hour_spike_that_the_mean_dilutes():
    ds = build_hourly_run(INIT, GRID, None, pull=_fake_pull_factory())
    spiked = ds.copy(deep=True)
    spiked["cape_max"].loc[dict(fxx=2)] = 1000.0  # a spike in the middle of bin 0

    out = aggregate_bin(spiked, 0)

    assert float(out["cape_max"].min()) == 1000.0
    assert float(aggregate_bin(spiked, 1)["cape_max"].max()) == 7.0  # bin 1 untouched


def test_process_one_hourly_run_writes_atomically_and_a_failed_hour_writes_nothing(tmp_path):
    staged = tmp_path / "staged"
    out = tmp_path / "hourly"
    _write_staged_v2_run(staged, INIT)

    def broken_pull(init_time, fxx, grid):
        if fxx == 6:
            raise RuntimeError("no idx")
        return _fake_pull_factory()(init_time, fxx, grid)

    ok, msg = process_one_hourly_run(INIT, GRID, staged, out, cache_root=None, pull=broken_pull)
    assert not ok and "RuntimeError" in msg
    assert not hourly_run_path(out, INIT).exists()
    assert not list(out.glob("*.partial.nc"))

    ok, _ = process_one_hourly_run(INIT, GRID, staged, out, cache_root=None, pull=_fake_pull_factory())
    assert ok and hourly_run_path(out, INIT).exists()
    assert not list(out.glob("*.partial.nc"))


def test_process_one_hourly_run_skips_an_already_staged_run(tmp_path):
    out = tmp_path / "hourly"
    calls = []
    process_one_hourly_run(INIT, GRID, None, out, None, pull=_fake_pull_factory(calls))
    first = len(calls)
    ok, _ = process_one_hourly_run(INIT, GRID, None, out, None, pull=_fake_pull_factory(calls))
    assert ok and len(calls) == first


def test_group_runs_by_date_keeps_same_date_runs_together_and_sorted():
    runs = [
        pd.Timestamp("2020-05-02 03:00"),
        pd.Timestamp("2020-05-01 20:00"),
        pd.Timestamp("2020-05-01 05:00"),
    ]
    groups = group_runs_by_date(runs)
    assert groups == [
        [pd.Timestamp("2020-05-01 05:00"), pd.Timestamp("2020-05-01 20:00")],
        [pd.Timestamp("2020-05-02 03:00")],
    ]


def test_derive_wholebin_run_matches_v2_staging_format_and_labels(tmp_path):
    hourly_dir = tmp_path / "hourly"
    ds = build_hourly_run(INIT, GRID, None, pull=_fake_pull_factory())
    hourly_dir.mkdir()
    ds.to_netcdf(hourly_run_path(hourly_dir, INIT))
    labels = _labels([("e1", INIT, 1, 5, 10), ("e2", INIT, 0, 20, 20), ("e3", INIT + pd.Timedelta(hours=1), 1, 30, 30)])

    out_dir = tmp_path / "wholebin"
    derive_wholebin_run(hourly_run_path(hourly_dir, INIT), labels, GRID, run_staging_path(out_dir, INIT))

    with xr.open_dataset(run_staging_path(out_dir, INIT)) as out:
        assert list(out.data_vars) == BASE_FEATURE_NAMES + ["label"]
        assert out.sizes["sample"] == 2
        assert list(out["bin_index"].values) == [0, 1]
        assert float(out["cape_mean"].isel(sample=1).mean()) == pytest.approx(5.5)
        assert int(out["label"].isel(sample=1).values[5, 10]) == 1
        assert int(out["label"].isel(sample=0).values[20, 20]) == 1
        assert int(out["label"].values.sum()) == 2  # e3 belongs to a different run


def test_derived_staging_is_readable_by_combine_staged_runs(tmp_path):
    hourly_dir, out_dir = tmp_path / "hourly", tmp_path / "wholebin"
    hourly_dir.mkdir()
    runs = [INIT, INIT + pd.Timedelta(hours=3)]
    for r in runs:
        build_hourly_run(r, GRID, None, pull=_fake_pull_factory()).to_netcdf(hourly_run_path(hourly_dir, r))

    manifest = derive_wholebin_staging(hourly_dir, out_dir, runs + [INIT + pd.Timedelta(days=9)], _labels([]), GRID)

    assert len(manifest) == 2  # the run with no hourly file is skipped, not an error
    assert len(load_manifest(out_dir / "manifest.json")) == 2
    combined = combine_staged_runs(out_dir, [(runs[0], 0), (runs[1], 1)])
    assert combined.sizes["sample"] == 2
    assert [v for v in combined.data_vars if v != "label"] == BASE_FEATURE_NAMES


def test_run_hourly_pull_skips_manifest_entries_and_retries_failed_only_on_request(tmp_path, monkeypatch):
    import tornado_predictor.build_hourly as bh

    done, failed, new = INIT, INIT + pd.Timedelta(days=1), INIT + pd.Timedelta(days=2)
    manifest_path = tmp_path / "manifest.json"
    bh.save_manifest(
        manifest_path,
        {done.isoformat(): {"status": "done", "message": "x"}, failed.isoformat(): {"status": "failed", "message": "boom"}},
    )
    seen = []

    def fake_group(runs, grid_path, staged_dir, out_dir, cache_root, feature_set="base"):
        seen.extend(runs)
        return [(r, True, "ok") for r in runs]

    monkeypatch.setattr(bh, "_process_date_group", fake_group)
    logs = []
    args = dict(
        grid_path=tmp_path / "g.nc", staged_dir=None, out_dir=tmp_path / "out",
        manifest_path=manifest_path, cache_root=None, log=logs.append,
    )

    manifest = run_hourly_pull([done, failed, new], **args)
    assert seen == [new]  # done skipped, failed not retried by default
    assert manifest[failed.isoformat()]["status"] == "failed"
    assert manifest[new.isoformat()]["status"] == "done"

    seen.clear()
    manifest = run_hourly_pull([done, failed, new], retry_failed=True, **args)
    assert seen == [failed]
    assert manifest[failed.isoformat()]["status"] == "done"
    assert load_manifest(manifest_path) == manifest


def _fake_extra_pull(init_time, fxx, grid):
    return {name: np.full(SHAPE, 100.0 + fxx, dtype="float32") for name in EXTRA_FEATURE_NAMES}


def test_extra_feature_names_are_distinct_from_base_and_cover_every_candidate_field():
    assert not set(EXTRA_FEATURE_NAMES) & set(BASE_FEATURE_NAMES)
    assert len(EXTRA_FEATURE_NAMES) == 16  # 9 pulls -> 8 pooled fields (shear u,v collapse) x (mean, max)
    for field in ("mlcape", "mlcin", "mucape", "mucin", "lcl_sb", "refc", "pres_sfc", "shear_0_1km"):
        assert f"{field}_mean" in EXTRA_FEATURE_NAMES and f"{field}_max" in EXTRA_FEATURE_NAMES
    assert FEATURE_SETS == {"base": BASE_FEATURE_NAMES, "extra": EXTRA_FEATURE_NAMES}


def test_extra_feature_set_pulls_all_eight_hours_and_ignores_staged_dir(tmp_path):
    _write_staged_v2_run(tmp_path / "staged", INIT)  # must NOT be used for the extras
    out = tmp_path / "extra"
    calls = []

    def pull(init_time, fxx, grid):
        calls.append(fxx)
        return _fake_extra_pull(init_time, fxx, grid)

    ok, _ = process_one_hourly_run(INIT, GRID, tmp_path / "staged", out, None, pull=pull, feature_set="extra")

    assert ok and sorted(calls) == list(HOURLY_FXX)
    with xr.open_dataset(hourly_run_path(out, INIT)) as ds:
        assert list(ds.data_vars) == EXTRA_FEATURE_NAMES
        assert ds.attrs["reused_fxx"] == ""
        assert float(ds["refc_mean"].sel(fxx=0).mean()) == 100.0  # pulled, not -1.0 from staging


def test_derive_wholebin_run_appends_extra_channels_after_the_base_14(tmp_path):
    hourly_dir, extra_dir, out_dir = tmp_path / "h", tmp_path / "e", tmp_path / "w"
    hourly_dir.mkdir(); extra_dir.mkdir()
    build_hourly_run(INIT, GRID, None, pull=_fake_pull_factory()).to_netcdf(hourly_run_path(hourly_dir, INIT))
    build_hourly_run(INIT, GRID, None, pull=_fake_extra_pull, feature_names=EXTRA_FEATURE_NAMES).to_netcdf(
        hourly_run_path(extra_dir, INIT)
    )

    derive_wholebin_run(
        hourly_run_path(hourly_dir, INIT), _labels([]), GRID, run_staging_path(out_dir, INIT),
        extra_hourly_path=hourly_run_path(extra_dir, INIT),
    )

    with xr.open_dataset(run_staging_path(out_dir, INIT)) as out:
        assert list(out.data_vars) == BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES + ["label"]
        assert float(out["pres_sfc_mean"].isel(sample=1).mean()) == pytest.approx(105.5)  # mean of 104..107
        assert float(out["pres_sfc_max"].isel(sample=0).mean()) == pytest.approx(103.0)


def test_derive_wholebin_staging_skips_runs_missing_their_extras_file(tmp_path):
    hourly_dir, extra_dir, out_dir = tmp_path / "h", tmp_path / "e", tmp_path / "w"
    hourly_dir.mkdir(); extra_dir.mkdir()
    runs = [INIT, INIT + pd.Timedelta(hours=1)]
    for r in runs:
        build_hourly_run(r, GRID, None, pull=_fake_pull_factory()).to_netcdf(hourly_run_path(hourly_dir, r))
    build_hourly_run(runs[0], GRID, None, pull=_fake_extra_pull, feature_names=EXTRA_FEATURE_NAMES).to_netcdf(
        hourly_run_path(extra_dir, runs[0])
    )

    manifest = derive_wholebin_staging(hourly_dir, out_dir, runs, _labels([]), GRID, extra_hourly_dir=extra_dir)

    assert list(manifest) == [runs[0].isoformat()]


def _patch_pulls(monkeypatch, nan_lcl_pixels=0, nan_cape_pixels=0, lcl_all_nan_cell=False):
    import tornado_predictor.build_hourly as bh
    from tornado_predictor.features import HRRR_NX, HRRR_NY

    def fake(init_time, fxx, search):
        a = np.full((HRRR_NY, HRRR_NX), 5.0, dtype="float32")
        if search.startswith("HGT:level of adiabatic"):
            a[0, :nan_lcl_pixels] = np.nan
            if lcl_all_nan_cell:
                a[: GRID.stride, : GRID.stride] = np.nan
        if search == "CAPE:180-0 mb above ground":
            a[0, :nan_cape_pixels] = np.nan
        return a

    monkeypatch.setattr(bh, "pull_hrrr_field", fake)
    return bh


def test_pull_extra_snapshot_pools_lcl_over_valid_pixels_only(monkeypatch):
    bh = _patch_pulls(monkeypatch, nan_lcl_pixels=5)  # a few undefined pixels in one cell
    snap = bh.pull_extra_snapshot(INIT, 0, GRID)
    assert list(snap) == EXTRA_FEATURE_NAMES
    assert not any(np.isnan(v).any() for v in snap.values())
    assert np.all(snap["lcl_sb_mean"] == 5.0) and np.all(snap["lcl_sb_max"] == 5.0)  # NaNs ignored, not zero-filled


def test_pull_extra_snapshot_still_rejects_nan_in_any_other_field(monkeypatch):
    bh = _patch_pulls(monkeypatch, nan_cape_pixels=1)
    with pytest.raises(ValueError, match="mlcape"):
        bh.pull_extra_snapshot(INIT, 0, GRID)


def test_pull_extra_snapshot_rejects_too_many_lcl_nans_and_fully_undefined_cells(monkeypatch):
    bh = _patch_pulls(monkeypatch, nan_lcl_pixels=2000)  # 2000 / 1.9M is under 1%: ok by the fraction alone
    bh.pull_extra_snapshot(INIT, 0, GRID)
    bh = _patch_pulls(monkeypatch, lcl_all_nan_cell=True)
    with pytest.raises(ValueError, match="no valid native pixel"):
        bh.pull_extra_snapshot(INIT, 0, GRID)
