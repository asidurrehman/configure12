"""
A tiny, made-up case study used by the tests.

It builds every input file that configure12_main.py expects under ``run_path``
(DEM, ponds grid, catchment, buildings and depth-damage tables) and provides
``fake_citycat``, a stand-in for ``citycat.exe``. The fake model writes a
``R{r}_C{c}_max_depth.csv`` file whose water depths fall as more pond storage
is dug into the DEM, so the optimisation has a genuine cost-damage trade-off.

None of the numbers here come from the published study; they only exercise
the code.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

CELL = 5            # DEM resolution in metres (pond grid cells are 5 m x 5 m = 25 m2)
NCOLS = 20
NROWS = 20
CRS = "EPSG:27700"  # British National Grid (any projected CRS would do)

# Three candidate ponds, each a rectangle of DEM cells: (col0, row0, ncols, nrows)
PONDS = {1: (2, 2, 6, 5),     # 30 cells = 750 m2 -> 7 size steps of 100 m2
         2: (12, 3, 5, 5),    # 25 cells = 625 m2 -> 6 size steps
         3: (4, 12, 4, 5)}    # 20 cells = 500 m2 -> 5 size steps

BUILDINGS = [  # (UID, Type, x0, y0)
    (1, "Residential", 30, 30),
    (2, "Residential", 60, 20),
    (3, "Retail", 40, 70),
    (4, "Offices", 70, 60),
    (5, "Public Buildings", 20, 80),
    (6, "Residential", 85, 85),
]


def dem_elevation(x, y):
    """Smooth synthetic terrain (metres) at coordinates x, y."""
    return 10.0 + 0.04 * x + 0.02 * y + 0.5 * np.sin(x / 17.0) * np.cos(y / 23.0)


def dem_array():
    """Return the synthetic DEM as a (rows, cols) array, top row first."""
    xs = (np.arange(NCOLS) + 0.5) * CELL
    ys = (NROWS - np.arange(NROWS) - 0.5) * CELL
    xx, yy = np.meshgrid(xs, ys)
    return np.round(dem_elevation(xx, yy), 2)


def write_dem(path):
    """Write the synthetic DEM as an ESRI ASCII grid with a 6-line header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ncols {NCOLS}\nnrows {NROWS}\nxllcorner 0\nyllcorner 0\n"
                f"cellsize {CELL}\nNODATA_value -9999\n")
        np.savetxt(f, dem_array(), fmt="%.2f")


def build_case(run_path):
    """Create all input files that configure12_main.py reads under run_path."""
    run_path = Path(run_path)
    vector = run_path / "vector"
    vector.mkdir(parents=True, exist_ok=True)

    write_dem(run_path / "domain_dem_original" / "Domain_DEM.asc")

    # Ponds grid: one 5 m x 5 m polygon per DEM cell, with pond Id and elevation
    rows = []
    for pond_id, (c0, r0, nc, nr) in PONDS.items():
        for c in range(c0, c0 + nc):
            for r in range(r0, r0 + nr):
                x0, y0 = c * CELL, r * CELL
                rows.append({"Id": pond_id,
                             "elevation": float(np.round(
                                 dem_elevation(x0 + CELL / 2, y0 + CELL / 2), 2)),
                             "geometry": box(x0, y0, x0 + CELL, y0 + CELL)})
    gpd.GeoDataFrame(rows, crs=CRS).to_file(vector / "all_possible_ponds_grid.shp")

    # Catchment (DEM extent)
    gpd.GeoDataFrame({"Id": [1]}, geometry=[box(0, 0, NCOLS * CELL, NROWS * CELL)],
                     crs=CRS).to_file(vector / "catchment_dem.shp")

    # Buildings with UID and Type fields
    gpd.GeoDataFrame(
        {"UID": [b[0] for b in BUILDINGS], "Type": [b[1] for b in BUILDINGS]},
        geometry=[box(x, y, x + 8, y + 8) for _, _, x, y in BUILDINGS],
        crs=CRS).to_file(vector / "buildings.shp")

    # Depth-damage tables (depth in metres; residential in GBP per building,
    # non-residential in GBP per square metre for retail, offices, public)
    dd = run_path / "dd_data"
    dd.mkdir(exist_ok=True)
    pd.DataFrame({"depth": [0.0, 0.1, 0.3, 0.6, 1.0, 3.0],
                  "damage": [0, 5000, 15000, 25000, 35000, 60000]}
                 ).to_csv(dd / "residential_damage_calculation.csv", index=False)
    pd.DataFrame({"depth": [0.0, 0.1, 0.3, 0.6, 1.0, 3.0],
                  "retail": [0, 50, 120, 200, 300, 500],
                  "offices": [0, 40, 100, 180, 260, 400],
                  "public": [0, 30, 90, 150, 220, 350]}
                 ).to_csv(dd / "non_residential_damage_calculation.csv", index=False)
    return run_path


_CMD = re.compile(r"cd (?P<path>.+?) & citycat\.exe -r (?P<r>\d+) -c (?P<c>\d+)")


def fake_citycat(cmd, shell=False, **kwargs):
    """Stand-in for ``subprocess.call('cd <run_path> & citycat.exe -r R -c C')``.

    Writes ``<run_path>/R{R}C{C}_SurfaceMaps/R{R}_C{C}_max_depth.csv`` with
    columns XCen, YCen, Depth for every DEM cell. Depth grows with the rainfall
    number and falls as more volume is dug out of ``<run_path>/Domain_DEM.asc``.
    """
    m = _CMD.fullmatch(cmd.strip())
    if m is None:
        raise AssertionError(f"unexpected CityCAT command: {cmd!r}")
    run_path, r, c = Path(m["path"]), int(m["r"]), int(m["c"])

    base = dem_array()
    modified = run_path / "Domain_DEM.asc"
    if modified.exists():
        dem = np.loadtxt(modified, skiprows=6)
        removed = float(np.sum(base - dem))   # metres of DEM lowered, summed over cells
    else:
        removed = 0.0

    xs = (np.arange(NCOLS) + 0.5) * CELL
    ys = (NROWS - np.arange(NROWS) - 0.5) * CELL
    xx, yy = np.meshgrid(xs, ys)
    depth = (0.15 + r / 20000.0 - 0.004 * removed
             + 0.25 * np.sin(xx / 15.0) * np.cos(yy / 20.0))
    depth = np.clip(np.round(depth, 3), 0, None)

    out_dir = run_path / f"R{r}C{c}_SurfaceMaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"XCen": xx.ravel(), "YCen": yy.ravel(), "Depth": depth.ravel()}
                 ).to_csv(out_dir / f"R{r}_C{c}_max_depth.csv", index=False)
    return 0


def prepare_ponds(ponds_grid_shp):
    """Apply the pond pre-processing done in configure12_main.py.

    Returns the processed ponds GeoDataFrame and the arrays that
    ``citycat_exposure`` and ``initial_pop`` need. Kept in step with the
    main script (same formulas, same constants).
    """
    min_elv = ponds_grid_shp.groupby('Id')['elevation'].min().reset_index()
    min_elv.rename(columns={'elevation': 'min_elv'}, inplace=True)
    ponds_grid_shp = ponds_grid_shp.merge(min_elv, on='Id')
    ponds_grid_shp["elv_diff"] = (ponds_grid_shp["min_elv"] -
                                  ponds_grid_shp["elevation"])
    ponds_grid_shp["pond_depth"] = -1
    ponds_grid_shp["pond_cell_total_depth"] = (ponds_grid_shp["elv_diff"] +
                                               ponds_grid_shp["pond_depth"])
    ponds_grid_shp['area'] = ponds_grid_shp.geometry.area
    ponds_grid_shp["lev_vol"] = (abs(ponds_grid_shp["elv_diff"]) *
                                 ponds_grid_shp['area'])
    ponds_stat = ponds_grid_shp.groupby('Id')['area'].sum().reset_index()
    ponds_stat['pond_depth'] = 1
    ponds_depth = np.array(ponds_stat['pond_depth'])[np.newaxis, :]
    ponds_stat['min_area'] = 100
    ponds_stat['area_steps'] = (ponds_stat['area'] /
                                ponds_stat['min_area']).astype(int)
    ponds_stat['gene_end'] = ponds_stat['area_steps'].cumsum()
    ponds_stat['gene_start'] = ponds_stat['gene_end'] - ponds_stat['area_steps']
    ponds_grid_cell_area = 25
    ponds_stat['area_step_val'] = ponds_stat['min_area'] / ponds_grid_cell_area
    return {
        "ponds_grid_shp": ponds_grid_shp,
        "ponds_depth": ponds_depth,
        "ponds_grid_cell_area": ponds_grid_cell_area,
        "ponds_area_step_val": np.array(ponds_stat['area_step_val'])[np.newaxis, :],
        "ponds_gene_start": np.array(ponds_stat['gene_start']),
        "ponds_gene_end": np.array(ponds_stat['gene_end']),
        "chrom_len_ponds_locat": ponds_grid_shp['Id'].nunique(),
        "chrom_len_ponds_size": int(ponds_stat['area_steps'].sum()),
    }


def write_config(path, run_path, pop_size=8, max_generations=2):
    """Write a config.yaml for the synthetic case (small, fast run)."""
    run = str(run_path).replace(os.sep, "/")
    Path(path).write_text(
        f"run_path: {run}\n"
        "rainfall_num: [1060, 2060, 3060, 5060, 10060]\n"
        "config_num: 3\n"
        f"pop_size: {pop_size}\n"
        f"max_generations: {max_generations}\n"
        "pond_unit_vol_lcc: 153.2\n"
        "plot_x_limit: [-0.05, 3.1]\n"
        "plot_y_limit: [5.84, 6.4]\n", encoding="utf-8")
    return Path(path)
