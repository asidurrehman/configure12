"""Tests for file reading and plotting helpers."""
import numpy as np


def test_read_dem_ascii(cf, tmp_path):
    path = tmp_path / "dem.asc"
    path.write_text("ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\n"
                    "cellsize 5\nNODATA_value -9999\n"
                    "1.00 2.00 3.00\n4.00 5.00 6.50\n")
    metadata, data = cf.read_dem_ascii(str(path))
    assert metadata == {"ncols": "3", "nrows": "2", "xllcorner": "100",
                        "yllcorner": "200", "cellsize": "5",
                        "NODATA_value": "-9999"}
    assert data.shape == (2, 3)
    assert np.array_equal(data, [[1, 2, 3], [4, 5, 6.5]])


def test_scatter_plot_writes_png(cf, tmp_path):
    out = tmp_path / "Generation_No_0"
    cf.scatter_plot("default", "Initial population", np.array([[0.1], [0.5]]),
                    np.array([[6.0], [5.9]]), "Evolving solution",
                    [-0.05, 3.1], [5.84, 6.4], "Cost", "Damage", str(out))
    png = tmp_path / "Generation_No_0.png"
    assert png.exists() and png.stat().st_size > 0
