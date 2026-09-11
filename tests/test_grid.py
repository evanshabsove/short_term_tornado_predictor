from shapely.geometry import Point

from tornado_predictor.grid import build_coarse_grid


def test_grid_shape():
    grid = build_coarse_grid()
    assert (grid.n_rows, grid.n_cols) == (81, 138)


def test_tangent_point_assigns_to_interior_cell():
    grid = build_coarse_grid()
    row, col = grid.assign_cells([-97.5], [38.5])
    assert row[0] != -1 and col[0] != -1
    assert 0 <= row[0] < grid.n_rows
    assert 0 <= col[0] < grid.n_cols


def test_point_far_outside_domain_is_out_of_bounds():
    grid = build_coarse_grid()
    row, col = grid.assign_cells([0.0], [0.0])
    assert (row[0], col[0]) == (-1, -1)


def test_cell_polygon_contains_its_own_center():
    grid = build_coarse_grid()
    for r, c in [(0, 0), (40, 70), (80, 137)]:
        poly = grid.cell_polygon(r, c)
        center = Point(grid.lon_center[r, c], grid.lat_center[r, c])
        assert poly.contains(center) or poly.touches(center)


def test_dataset_roundtrip():
    grid = build_coarse_grid()
    rebuilt = type(grid).from_dataset(grid.to_dataset())
    assert (rebuilt.lat_center == grid.lat_center).all()
    assert (rebuilt.lon_center == grid.lon_center).all()
    assert rebuilt.cell_size_m == grid.cell_size_m
