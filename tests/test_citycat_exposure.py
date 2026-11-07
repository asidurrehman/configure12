"""Tests for spatial_indices and citycat_exposure using a fake CityCAT.

The real CityCAT is replaced by synthetic_case.fake_citycat, so these tests
check the data handling around the model (pond placement, DEM editing,
exposure classes, damage and cost), not the hydrodynamics.
"""
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest

import synthetic_case as sc

RAINFALL_NUM = np.array([1060, 2060, 3060, 5060, 10060])
LCC = 153.2  # GBP per m3


@pytest.fixture
def case(cf, tmp_path, fake_citycat):
    run_path = sc.build_case(tmp_path / "run")
    rp = str(run_path)
    bldg_file = os.path.join(rp, "vector", "buildings.shp")
    csv_file = os.path.join(rp, "R1060C1_SurfaceMaps", "R1060_C1_max_depth.csv")
    shp_field, cell_index, buffer_list = cf.spatial_indices(
        bldg_file, rp, 1060, 1, csv_file)
    ponds = sc.prepare_ponds(
        gpd.read_file(os.path.join(rp, "vector", "all_possible_ponds_grid.shp")))
    dem_metadata, dem_data = cf.read_dem_ascii(
        os.path.join(rp, "domain_dem_original", "Domain_DEM.asc"))
    return dict(
        run_path=rp, bldg_file=bldg_file, shp_field=shp_field,
        cell_index=cell_index, buffer_list=buffer_list, ponds=ponds,
        catch_shp=gpd.read_file(os.path.join(rp, "vector", "catchment_dem.shp")),
        dem_metadata=dem_metadata, dem_data=dem_data,
        res_dd=np.array(pd.read_csv(os.path.join(
            rp, "dd_data", "residential_damage_calculation.csv"))),
        nonres_dd=np.array(pd.read_csv(os.path.join(
            rp, "dd_data", "non_residential_damage_calculation.csv"))),
    )


def run_exposure(cf, case, size_val):
    p = case["ponds"]
    size_val = np.asarray(size_val, dtype=float)
    return cf.citycat_exposure(
        case["bldg_file"], p["chrom_len_ponds_locat"], len(size_val),
        case["run_path"], RAINFALL_NUM, 3, size_val, p["ponds_grid_shp"],
        p["ponds_depth"], p["ponds_grid_cell_area"], p["ponds_area_step_val"],
        LCC, case["catch_shp"], case["dem_metadata"], case["dem_data"],
        case["shp_field"], case["cell_index"], case["buffer_list"], 0,
        case["res_dd"], case["nonres_dd"])


def test_spatial_indices(case):
    assert case["shp_field"] == "UID"
    assert len(case["cell_index"]) == len(case["buffer_list"]) > 0
    assert set(case["buffer_list"]) == {b[0] for b in sc.BUILDINGS}
    assert 0 <= min(case["cell_index"])
    assert max(case["cell_index"]) < sc.NCOLS * sc.NROWS


def test_citycat_exposure(cf, case):
    # no ponds, every pond at full size (7, 6, 5 steps), and a partial design
    volume, cost, high, medium, low, damage = run_exposure(
        cf, case, [[0, 0, 0], [7, 6, 5], [3, 0, 2]])

    assert volume.shape == (3, 3) and cost.shape == (3, 1)
    for arr in (high, medium, low, damage):
        assert arr.shape == (3, len(RAINFALL_NUM))

    # one size step = 100 m2 (4 cells of 25 m2) dug 1 m deep
    assert np.array_equal(volume, [[0, 0, 0], [700, 600, 500], [300, 0, 200]])

    # life cycle cost in million GBP, rounded per pond and in total
    expected = [round(np.sum(np.round(v * LCC / 1e6, 3)), 3) for v in volume]
    assert np.allclose(cost[:, 0], expected)
    assert cost[0, 0] == 0

    # every building is in exactly one exposure class
    assert np.all(high + medium + low == len(sc.BUILDINGS))

    # damage is never negative, grows with rainfall when there are no ponds,
    # and the fake model makes full-size ponds reduce it
    assert np.all(damage >= 0)
    assert np.all(np.diff(damage[0]) >= 0)
    assert damage[0, -1] > 0
    assert np.all(damage[1] <= damage[0])


def test_no_ponds_leaves_dem_unchanged(cf, case):
    run_exposure(cf, case, [[0, 0, 0]])
    written = np.loadtxt(os.path.join(case["run_path"], "Domain_DEM.asc"),
                         skiprows=6)
    assert np.allclose(written, case["dem_data"])


def test_ponds_are_dug_into_the_dem(cf, case):
    run_exposure(cf, case, [[7, 0, 0]])
    written = np.loadtxt(os.path.join(case["run_path"], "Domain_DEM.asc"),
                         skiprows=6)
    lowered = case["dem_data"] - written
    assert np.all(lowered >= -1e-9)
    # pond 1 at full size covers 28 of its 30 cells
    assert np.count_nonzero(lowered > 1e-9) == 28
