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


def test_vertex_lattice_reproduces_every_cell_polygon():
    import numpy as np

    grid = build_coarse_grid()
    lon, lat = grid.vertex_lonlat()
    assert lon.shape == lat.shape == (grid.n_rows + 1, grid.n_cols + 1)
    worst_m = 0.0
    for row in range(0, grid.n_rows, 3):  # every third row keeps the test quick; all columns
        for col in range(grid.n_cols):
            poly = np.array(grid.cell_polygon(row, col).exterior.coords)[:4]  # SW, SE, NE, NW
            corners = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)]
            for (vi, vj), (plon, plat) in zip(corners, poly):
                assert abs(lon[vi, vj] - plon) < 1e-7 and abs(lat[vi, vj] - plat) < 1e-7
                # what the API serves: 4 decimals; the error must stay under ~7.5 m
                dlat = (round(lat[vi, vj], 4) - plat) * 111_195
                dlon = (round(lon[vi, vj], 4) - plon) * 111_195 * np.cos(np.radians(plat))
                worst_m = max(worst_m, float(np.hypot(dlat, dlon)))
    assert worst_m < 7.5  # measured worst case over all vertices: 7.4 m (5.6 m per axis)


def test_vertex_lattice_is_ordered_south_to_north_and_west_to_east():
    grid = build_coarse_grid()
    lon, lat = grid.vertex_lonlat()
    assert lat[1, 0] > lat[0, 0] and lon[0, 1] > lon[0, 0]  # row 0 is the southernmost, col 0 the westernmost
