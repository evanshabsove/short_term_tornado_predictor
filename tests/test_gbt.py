import numpy as np
import pytest
import xarray as xr

from tornado_predictor.gbt import downsample_negative_cells, flatten_for_gbt


def _tiny_dataset(n_samples=3, n_rows=4, n_cols=5, n_positive_cells=6, seed=0):
    rng = np.random.default_rng(seed)
    label = np.zeros((n_samples, n_rows, n_cols), dtype=np.int8)
    flat_label = label.reshape(-1)
    positive_flat_idx = rng.choice(flat_label.size, size=n_positive_cells, replace=False)
    flat_label[positive_flat_idx] = 1
    label = flat_label.reshape(n_samples, n_rows, n_cols)

    ds = xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), rng.normal(size=(n_samples, n_rows, n_cols)).astype("float32")),
            "cape_max": (("sample", "row", "col"), rng.normal(size=(n_samples, n_rows, n_cols)).astype("float32")),
            "label": (("sample", "row", "col"), label),
        },
        coords={"sample": np.arange(n_samples), "row": np.arange(n_rows), "col": np.arange(n_cols)},
    )
    return ds, n_positive_cells, n_samples * n_rows * n_cols - n_positive_cells


# --- flatten_for_gbt ---


def test_flatten_for_gbt_shapes():
    ds, n_pos, n_neg = _tiny_dataset()
    X, y, feature_names = flatten_for_gbt(ds)

    assert feature_names == ["cape_mean", "cape_max"]
    assert X.shape == (3 * 4 * 5, 2)
    assert y.shape == (3 * 4 * 5,)
    assert y.sum() == n_pos


def test_flatten_for_gbt_preserves_row_alignment_between_X_and_y():
    """A positive cell's label (y==1) must line up with that exact
    cell's feature row in X, not some other cell after flattening."""
    ds = xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype="float32")),
            "label": (("sample", "row", "col"), np.array([[[0, 1], [0, 0]]], dtype=np.int8)),
        },
        coords={"sample": [0], "row": [0, 1], "col": [0, 1]},
    )
    X, y, _ = flatten_for_gbt(ds)
    positive_row_value = X[y == 1][0, 0]
    assert positive_row_value == 2.0  # the cell with label=1 has cape_mean=2.0


# --- downsample_negative_cells ---


def test_downsample_negative_cells_keeps_every_positive():
    ds, n_pos, n_neg = _tiny_dataset(n_samples=5, n_rows=10, n_cols=10, n_positive_cells=20)
    X, y, _ = flatten_for_gbt(ds)

    X_sub, y_sub = downsample_negative_cells(X, y, negative_ratio=5, seed=0)

    assert y_sub.sum() == n_pos  # every positive row kept
    assert (y_sub == 0).sum() == 5 * n_pos  # exact requested ratio
    assert len(y_sub) == n_pos + 5 * n_pos


def test_downsample_negative_cells_is_reproducible_given_same_seed():
    ds, _, _ = _tiny_dataset(n_samples=5, n_rows=10, n_cols=10, n_positive_cells=20)
    X, y, _ = flatten_for_gbt(ds)

    X1, y1 = downsample_negative_cells(X, y, negative_ratio=3, seed=42)
    X2, y2 = downsample_negative_cells(X, y, negative_ratio=3, seed=42)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_downsample_negative_cells_different_seeds_give_different_samples():
    ds, _, _ = _tiny_dataset(n_samples=5, n_rows=10, n_cols=10, n_positive_cells=20)
    X, y, _ = flatten_for_gbt(ds)

    X1, _ = downsample_negative_cells(X, y, negative_ratio=3, seed=0)
    X2, _ = downsample_negative_cells(X, y, negative_ratio=3, seed=1)

    assert not np.array_equal(X1, X2)


def test_downsample_negative_cells_raises_if_not_enough_negatives():
    ds, n_pos, n_neg = _tiny_dataset(n_samples=1, n_rows=3, n_cols=3, n_positive_cells=8)  # 8 pos, 1 neg
    X, y, _ = flatten_for_gbt(ds)

    with pytest.raises(ValueError):
        downsample_negative_cells(X, y, negative_ratio=20, seed=0)  # wants 160 negatives, only 1 exists
