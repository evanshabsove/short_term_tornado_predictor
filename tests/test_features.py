import numpy as np
import pandas as pd

from tornado_predictor.features import (
    FIELD_SPECS,
    FIELD_UNITS,
    NEW_FIELD_NAMES_2026_09,
    OPTIONAL_FIELD_NAMES,
    UH_EXTRA_LAYERS_AVAILABLE_FROM,
    pool_field_to_grid,
    uh_layers_available_value,
)
from tornado_predictor.grid import HRRR_NX, HRRR_NY, build_coarse_grid

GRID = build_coarse_grid()


def test_updraft_helicity_fields_have_exact_search_strings():
    """Regression guard: these exact search strings were verified live
    against a real HRRR .idx file (twice, at two different dates/fxx) --
    a typo here (e.g. wrong level-text ordering) would silently pull
    the wrong field or fail to match at all. A loose "MXUPHL:" search
    would also be ambiguous across the 0-2km/0-3km/2-5km layers, same
    concern already noted for "CAPE:" in this module's docstring."""
    specs = dict(FIELD_SPECS)
    assert specs["uh_0_2km"] == "MXUPHL:2000-0 m above ground"
    assert specs["uh_0_3km"] == "MXUPHL:3000-0 m above ground"
    assert specs["uh_2_5km"] == "MXUPHL:5000-2000 m above ground"


def test_new_field_names_are_all_present_in_field_specs():
    field_names = {name for name, _ in FIELD_SPECS}
    for name in NEW_FIELD_NAMES_2026_09:
        assert name in field_names
        assert name in FIELD_UNITS


def test_optional_field_names_are_the_two_partial_coverage_uh_layers():
    """uh_2_5km is deliberately NOT optional -- verified live to be
    available across the full 2014-2025 archive, unlike the other two."""
    assert OPTIONAL_FIELD_NAMES == {"uh_0_2km", "uh_0_3km"}
    assert "uh_2_5km" not in OPTIONAL_FIELD_NAMES


def test_uh_layers_available_value_before_and_after_boundary():
    assert uh_layers_available_value("2018-07-10") == 0.0
    assert uh_layers_available_value("2018-07-13") == 1.0
    assert uh_layers_available_value("2014-09-01") == 0.0
    assert uh_layers_available_value("2021-12-10") == 1.0
    assert uh_layers_available_value(UH_EXTRA_LAYERS_AVAILABLE_FROM) == 1.0
    assert uh_layers_available_value(UH_EXTRA_LAYERS_AVAILABLE_FROM - pd.Timedelta(seconds=1)) == 0.0


def test_constant_field_pools_to_same_constant():
    field = np.full((HRRR_NY, HRRR_NX), 7.5)
    mean, max_ = pool_field_to_grid(field, GRID)
    assert mean.shape == (GRID.n_rows, GRID.n_cols)
    assert max_.shape == (GRID.n_rows, GRID.n_cols)
    assert np.allclose(mean, 7.5)
    assert np.allclose(max_, 7.5)


def test_spike_is_captured_by_max_but_diluted_in_mean():
    field = np.zeros((HRRR_NY, HRRR_NX))
    stride = GRID.stride
    # Put a spike at the first native pixel of block (2, 3).
    field[2 * stride, 3 * stride] = 1000.0
    mean, max_ = pool_field_to_grid(field, GRID)

    assert max_[2, 3] == 1000.0
    assert mean[2, 3] == 1000.0 / (stride * stride)

    # Neighboring blocks are unaffected.
    assert max_[2, 4] == 0.0
    assert mean[2, 4] == 0.0
    assert max_[1, 3] == 0.0


def test_mean_never_exceeds_max():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(HRRR_NY, HRRR_NX))
    mean, max_ = pool_field_to_grid(field, GRID)
    assert (mean <= max_ + 1e-12).all()
