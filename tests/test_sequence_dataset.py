from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tornado_predictor.build_hourly import (
    BASE_FEATURE_NAMES,
    aggregate_bin,
    build_hourly_run,
    derive_wholebin_run,
    hourly_run_path,
)
from tornado_predictor.build_scaled import load_manifest, run_staging_path
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.sequence_dataset import DenseSequenceDataset, flatten_run_bins, runs_from_run_bins
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS, fxx_values_for_bin
from tornado_predictor.training import QuietMapDownsampler

GRID = build_coarse_grid()
SHAPE = (GRID.n_rows, GRID.n_cols)
REPO = Path(__file__).resolve().parents[1]
HOURLY = REPO / "data/interim/scaled_build_hourly"
WHOLEBIN = REPO / "data/interim/scaled_build_wholebin"
V2 = REPO / "data/interim/scaled_build_v2"
SCORES = REPO / "data/interim/diag_scores"
UNET_CKPT = REPO / "models/unet/tornado_unet.pt"

needs_real_data = pytest.mark.skipif(
    not (HOURLY.exists() and WHOLEBIN.exists()), reason="hourly / whole-bin staging not built"
)

ACTIVE_INIT = pd.Timestamp("2021-12-10 21:00:00")
QUIET_INIT = pd.Timestamp("2021-12-10 22:00:00")


def _labels_df(rows):
    df = pd.DataFrame(rows, columns=["event_id", "init_time", "bin_index", "row", "col"])
    return df.astype({"bin_index": "int64", "row": "int64", "col": "int64", "init_time": "datetime64[ns]"})


def _fake_pull(init_time, fxx, grid):
    # value depends on fxx AND on the channel, so a swapped time or channel axis is visible
    return {name: np.full(SHAPE, 100.0 * i + fxx, dtype="float32") for i, name in enumerate(BASE_FEATURE_NAMES)}


@pytest.fixture
def synthetic(tmp_path):
    """Two runs: `ACTIVE_INIT` with one positive cell in bin 1 only, `QUIET_INIT` with none."""
    hourly, wholebin = tmp_path / "hourly", tmp_path / "wholebin"
    hourly.mkdir()
    labels = _labels_df([("ev", ACTIVE_INIT, 1, 5, 7)])
    for init in (ACTIVE_INIT, QUIET_INIT):
        ds = build_hourly_run(init, GRID, None, pull=_fake_pull)
        ds.to_netcdf(hourly_run_path(hourly, init))
        derive_wholebin_run(hourly_run_path(hourly, init), labels, GRID, run_staging_path(wholebin, init))
    return hourly, wholebin


def test_item_shapes_dtypes_and_axis_order(synthetic):
    hourly, wholebin = synthetic
    ds = DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT, QUIET_INIT])
    x, y = ds[0]
    assert x.shape == (8, 14, *SHAPE) and x.dtype == np.float32
    assert y.shape == (N_BINS, *SHAPE) and y.dtype == np.float32
    # axis 0 is fxx, axis 1 is the channel, in BASE_FEATURE_NAMES order
    for fxx in range(8):
        for i in range(14):
            assert x[fxx, i, 0, 0] == 100.0 * i + fxx
    assert ds.feature_names == BASE_FEATURE_NAMES


def test_labels_are_per_bin_and_run_level_activity_uses_either_bin(synthetic):
    hourly, wholebin = synthetic
    ds = DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT, QUIET_INIT])
    _, y = ds[0]
    assert y[0].sum() == 0 and y[1].sum() == 1 and y[1, 5, 7] == 1
    assert ds.is_active_by_bin.tolist() == [[False, True], [False, False]]
    assert ds.is_active.tolist() == [True, False]  # active if EITHER bin is; quiet only if BOTH are


def test_quiet_downsampler_works_on_run_level_flags(synthetic):
    hourly, wholebin = synthetic
    ds = DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT, QUIET_INIT])
    kept = list(QuietMapDownsampler(ds.is_active, quiet_keep_fraction=0.0, seed=0))
    assert kept == [0]  # every active run, no quiet ones


def test_feature_subset_follows_the_requested_order(synthetic):
    hourly, wholebin = synthetic
    names = ["shear_0_6km_max", "cape_mean"]
    ds = DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT], feature_names=names)
    x, _ = ds[0]
    assert x.shape[1] == 2
    assert x[0, 0, 0, 0] == 100.0 * BASE_FEATURE_NAMES.index("shear_0_6km_max")
    assert x[0, 1, 0, 0] == 100.0 * BASE_FEATURE_NAMES.index("cape_mean")


def test_non_finite_inputs_raise_instead_of_being_imputed(synthetic):
    hourly, wholebin = synthetic
    path = hourly_run_path(hourly, ACTIVE_INIT)
    with xr.open_dataset(path) as d:
        bad = d.load()
    bad["cape_mean"].values[3, 2, 2] = np.nan
    path.unlink()
    bad.to_netcdf(path)
    ds = DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT])
    with pytest.raises(ValueError, match="non-finite"):
        ds[0]


def test_missing_hourly_file_and_duplicate_runs_raise(synthetic):
    hourly, wholebin = synthetic
    with pytest.raises(FileNotFoundError):
        DenseSequenceDataset(hourly, wholebin, [pd.Timestamp("2000-01-01 00:00")])
    with pytest.raises(ValueError, match="duplicate"):
        DenseSequenceDataset(hourly, wholebin, [ACTIVE_INIT, ACTIVE_INIT])


def test_runs_from_run_bins_keeps_first_appearance_order():
    a, b = pd.Timestamp("2020-05-01 03:00"), pd.Timestamp("2020-04-01 03:00")
    assert runs_from_run_bins([(a, 0), (a, 1), (b, 0), (b, 1)]) == [a, b]


def test_flatten_run_bins_is_run_major_with_bin_fastest():
    arr = np.arange(3 * N_BINS * 2).reshape(3, N_BINS, 2)
    flat = flatten_run_bins(arr)
    assert flat.shape == (3 * N_BINS, 2)
    # sample 2k+b is (run k, bin b): the order combine_staged_runs gives arm A
    assert np.array_equal(flat[3], arr[1, 1]) and np.array_equal(flat[4], arr[2, 0])
    with pytest.raises(ValueError):
        flatten_run_bins(np.zeros((3, 5, 2)))


# ---- against the real staged data --------------------------------------------------


def _first_run_with_label(want_active: bool):
    manifest = load_manifest(WHOLEBIN / "manifest.json")
    for key in sorted(manifest):
        with xr.open_dataset(run_staging_path(WHOLEBIN, pd.Timestamp(key))) as w:
            if bool((w["label"].values.sum() > 0)) == want_active:
                return pd.Timestamp(key)


@needs_real_data
def test_real_run_inputs_reproduce_the_wholebin_features_and_labels():
    """The hourly inputs a sequence model sees must aggregate to exactly the whole-bin
    features arm A trained on, and the labels must be the whole-bin labels."""
    run = _first_run_with_label(True)
    ds = DenseSequenceDataset(HOURLY, WHOLEBIN, [run])
    x, y = ds[0]
    assert x.shape == (8, 14, *SHAPE) and np.isfinite(x).all()
    hourly = xr.Dataset({n: (("fxx", "row", "col"), x[:, i]) for i, n in enumerate(BASE_FEATURE_NAMES)},
                        coords={"fxx": list(range(8))})
    with xr.open_dataset(run_staging_path(WHOLEBIN, run)) as w:
        for b in range(N_BINS):
            agg = aggregate_bin(hourly, b)
            for name in BASE_FEATURE_NAMES:
                np.testing.assert_allclose(agg[name].values, w[name].isel(sample=b).values, rtol=1e-6)
        np.testing.assert_array_equal(y, w["label"].values.astype("float32"))
    assert y.sum() > 0 and ds.is_active[0]


@needs_real_data
@pytest.mark.skipif(not V2.exists(), reason="v2 staging not present")
def test_real_run_fxx_0_and_4_equal_the_v2_single_snapshots():
    run = _first_run_with_label(True)
    x, _ = DenseSequenceDataset(HOURLY, WHOLEBIN, [run])[0]
    with xr.open_dataset(run_staging_path(V2, run)) as v2:
        for fxx, sample in ((0, 0), (4, 1)):
            for i, name in enumerate(BASE_FEATURE_NAMES):
                np.testing.assert_allclose(x[fxx, i], v2[name].isel(sample=sample).values, rtol=1e-6)


@needs_real_data
def test_real_quiet_run_is_flagged_quiet():
    run = _first_run_with_label(False)
    ds = DenseSequenceDataset(HOURLY, WHOLEBIN, [run])
    assert not ds.is_active[0] and ds.y.sum() == 0


@needs_real_data
@pytest.mark.skipif(not UNET_CKPT.exists(), reason="U-Net checkpoint not present")
def test_channel_order_equals_the_14_feature_checkpoint_order():
    import torch

    ckpt = torch.load(UNET_CKPT, map_location="cpu", weights_only=False)
    assert list(ckpt["feature_names"]) == BASE_FEATURE_NAMES  # the dataset's default channel order


@needs_real_data
@pytest.mark.skipif(not (SCORES / "meta_2020.npz").exists(), reason="cached baseline scores not present")
def test_fold_2020_test_sample_order_matches_arm_a_after_flattening():
    manifest = load_manifest(WHOLEBIN / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in manifest.items() if v["status"] == "done" for b in range(N_BINS)]
    _, test_runs = split_run_bins_year_holdout(run_bins, 2020, buffer_hours=24.0)
    runs = runs_from_run_bins(test_runs)
    ds = DenseSequenceDataset(HOURLY, WHOLEBIN, runs)
    meta = np.load(SCORES / "meta_2020.npz")
    init = np.repeat(np.array([r.to_datetime64() for r in runs]), N_BINS).astype("datetime64[s]").astype("int64")
    bins = np.tile(np.arange(N_BINS), len(runs))
    assert np.array_equal(init, meta["init_time"]) and np.array_equal(bins, meta["bin_index"])
    # flattened labels line up with the (run, bin) sample count A scored
    assert flatten_run_bins(ds.y).shape[0] == len(meta["init_time"])
