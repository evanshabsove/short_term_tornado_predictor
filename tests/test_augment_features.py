import numpy as np
import xarray as xr

from tornado_predictor.augment_features import merge_new_fields

# merge_new_fields is the pure, offline-testable part of this module.
# augment_one_run itself makes live Herbie calls, same as
# build_scaled.process_one_run/finalize -- verified via a real smoke
# test against actual staged files (see CLAUDE.md), not pytest, matching
# this project's existing convention of not mocking network calls in
# the automated suite.


def _tiny_staged_dataset(n_samples=2, n_rows=4, n_cols=5):
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), rng.normal(size=(n_samples, n_rows, n_cols)).astype("float32")),
            "cape_max": (("sample", "row", "col"), rng.normal(size=(n_samples, n_rows, n_cols)).astype("float32")),
            "label": (("sample", "row", "col"), np.zeros((n_samples, n_rows, n_cols), dtype="int8")),
        },
        coords={
            "row": np.arange(n_rows),
            "col": np.arange(n_cols),
            "init_time": ("sample", [np.datetime64("2021-12-10T21:00")] * n_samples),
            "bin_index": ("sample", list(range(n_samples))),
        },
    )


def test_merge_new_fields_adds_mean_max_variables():
    ds = _tiny_staged_dataset()
    mean_stack = np.ones((2, 4, 5), dtype="float32") * 7.0
    max_stack = np.ones((2, 4, 5), dtype="float32") * 9.0

    merged = merge_new_fields(ds, {"uh_2_5km": (mean_stack, max_stack)})

    assert "uh_2_5km_mean" in merged.data_vars
    assert "uh_2_5km_max" in merged.data_vars
    assert merged["uh_2_5km_mean"].dims == ("sample", "row", "col")
    assert np.allclose(merged["uh_2_5km_mean"].values, 7.0)
    assert np.allclose(merged["uh_2_5km_max"].values, 9.0)
    assert merged["uh_2_5km_mean"].attrs["units"] == "m2 s-2"


def test_merge_new_fields_preserves_existing_data_unchanged():
    ds = _tiny_staged_dataset()
    original_cape_mean = ds["cape_mean"].values.copy()
    original_label = ds["label"].values.copy()

    mean_stack = np.zeros((2, 4, 5), dtype="float32")
    max_stack = np.zeros((2, 4, 5), dtype="float32")
    merged = merge_new_fields(ds, {"uh_0_2km": (mean_stack, max_stack)})

    assert np.array_equal(merged["cape_mean"].values, original_cape_mean)
    assert np.array_equal(merged["label"].values, original_label)
    assert list(merged["init_time"].values) == list(ds["init_time"].values)
    assert list(merged["bin_index"].values) == list(ds["bin_index"].values)


def test_merge_new_fields_adds_multiple_fields_independently():
    ds = _tiny_staged_dataset()
    new_fields = {
        "uh_0_2km": (np.full((2, 4, 5), 1.0, dtype="float32"), np.full((2, 4, 5), 2.0, dtype="float32")),
        "uh_0_3km": (np.full((2, 4, 5), 3.0, dtype="float32"), np.full((2, 4, 5), 4.0, dtype="float32")),
        "uh_2_5km": (np.full((2, 4, 5), 5.0, dtype="float32"), np.full((2, 4, 5), 6.0, dtype="float32")),
    }

    merged = merge_new_fields(ds, new_fields)

    for name, (mean_val, max_val) in [("uh_0_2km", (1.0, 2.0)), ("uh_0_3km", (3.0, 4.0)), ("uh_2_5km", (5.0, 6.0))]:
        assert np.allclose(merged[f"{name}_mean"].values, mean_val)
        assert np.allclose(merged[f"{name}_max"].values, max_val)

    # original 2 data vars (cape_mean, cape_max) + label + 3 new fields x 2 stats = 9 total
    assert len(merged.data_vars) == 9


def test_merge_new_fields_passes_through_nan_for_optional_fields():
    """The whole point of NaN handling: an unavailable optional field
    (e.g. 0-2km UH before 2018-07-13) merges in as real NaN values, not
    an error and not silently zeroed -- imputation happens later, at
    training.DenseGridDataset load time, not here."""
    ds = _tiny_staged_dataset()
    nan_stack = np.full((2, 4, 5), np.nan, dtype="float32")

    merged = merge_new_fields(ds, {"uh_0_2km": (nan_stack, nan_stack)})

    assert np.isnan(merged["uh_0_2km_mean"].values).all()
    assert np.isnan(merged["uh_0_2km_max"].values).all()
