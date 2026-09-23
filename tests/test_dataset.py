import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.dataset import label_grid_for_sample, representative_fxx, to_training_arrays
from tornado_predictor.grid import build_coarse_grid

GRID = build_coarse_grid()


def test_representative_fxx_matches_bin_start():
    assert representative_fxx(0) == 0
    assert representative_fxx(1) == 4


def test_label_grid_for_sample_scatters_matching_rows_only():
    init_time = pd.Timestamp("2021-12-10 22:00:00")
    labels_df = pd.DataFrame(
        [
            {"event_id": "e1", "ef_rating": 1.0, "init_time": init_time, "bin_index": 1, "row": 5, "col": 10},
            {"event_id": "e1", "ef_rating": 1.0, "init_time": init_time, "bin_index": 1, "row": 5, "col": 11},
            # different bin -- must not appear in bin_index=1's label grid
            {"event_id": "e2", "ef_rating": 2.0, "init_time": init_time, "bin_index": 0, "row": 20, "col": 20},
            # different init_time -- must not appear either
            {"event_id": "e3", "ef_rating": 0.0, "init_time": pd.Timestamp("2021-12-10 21:00:00"), "bin_index": 1, "row": 30, "col": 30},
        ]
    )
    label = label_grid_for_sample(labels_df, init_time, 1, GRID)

    assert label.shape == (GRID.n_rows, GRID.n_cols)
    assert label.sum() == 2
    assert label[5, 10] == 1 and label[5, 11] == 1
    assert label[20, 20] == 0
    assert label[30, 30] == 0


def test_to_training_arrays_shape_and_order():
    ds = xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), np.zeros((2, 3, 4))),
            "cape_max": (("sample", "row", "col"), np.ones((2, 3, 4))),
            "label": (("sample", "row", "col"), np.zeros((2, 3, 4), dtype=np.int8)),
        },
        coords={"sample": [0, 1], "row": [0, 1, 2], "col": [0, 1, 2, 3]},
    )
    X, y, feature_names = to_training_arrays(ds)

    assert feature_names == ["cape_mean", "cape_max"]
    assert X.shape == (2, 12, 2)
    assert y.shape == (2, 12)
    assert np.allclose(X[..., 0], 0.0)
    assert np.allclose(X[..., 1], 1.0)
