# -*- coding: utf-8 -*-
"""
╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝

File name: configure12_functions.py

@author: Asid Ur Rehman

Exposure analysis adapted from Robert Berstch
(https://doi.org/10.1016/j.envsoft.2022.105490)

Organisation: Newcastle University

About
------
This script contains all function definitions used to run
CONFIGURE (Cost benefit optimisation Framework for Implementing blue Green
infrastructure), a framework that integrates an evolutionary genetic algorithm
(NSGA II) with CityCAT to find optimal locations and sizes of BGI for
cost effective deployment. This version is designed specifically to optimise
the location and size of detention ponds for multiple rainfall return periods
by calculating Expected Annual Damage (EAD) using Direct Damage Cost (DDC)
for each return period.
"""
import os
import shutil  # must be imported before GDAL
import rtree
import re
from shapely.geometry import shape, Point
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import matplotlib.pyplot as plt
import warnings
from operator import itemgetter
import copy
from rasterio.features import rasterize
from rasterio.transform import from_origin

warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------#
# OVERVIEW OF THIS MODULE
#-----------------------------------------------------------------------------#
# All functions used by configure12_main.py (see HOW THIS SCRIPT WORKS there).
#   Flood model and damage:
#     spatial_indices()     one-off CityCAT run linking buildings to grid cells
#     read_dem_ascii()      read the original DEM
#     citycat_exposure()    dig the ponds into the DEM, run CityCAT for every
#                           rainfall, then work out exposure, damage and cost
#     auc_func()            area under the cost-damage curve (stopping rule)
#   NSGA-II genetic algorithm:
#     initial_pop()                         random initial designs
#     non_dominated_sorting()               rank designs into Pareto fronts
#     calculate_crowding_distance()         how isolated a design is in its front
#     fitness_sort()                        order designs from fittest to least fit
#     fitter_parent()                       binary tournament selection
#     crossover_uniform_mutation_bitflip()  create two children
#     remove_duplicate_list(), remove_duplicate_same_population(),
#     remove_duplicate_ponds_settigs(), remove_same_objectives_population(),
#     separate_new_old()                    bookkeeping of repeated designs
#   Plotting: scatter_plot()
# Arguments start with f_ (function argument). The three forms of a pond design
# (location chromosome, size chromosome, size values) are explained under
# NAMING CONVENTIONS in configure12_main.py.
#
###############################################################################
""" Functions section starts here """
###############################################################################
    
#-----------------------------------------------------------------------------#
# Run CityCAT to generate one time spatial indices for exposure analysis
#-----------------------------------------------------------------------------#
def spatial_indices(f_bldg_file, f_run_path, f_idx_rainfall, f_idx_config,
                    f_csv_file):
    """Run CityCAT once and link each building to the grid cells around it.

    CityCAT is run for rainfall ``f_idx_rainfall`` and configuration
    ``f_idx_config`` so that its maximum-depth CSV (``f_csv_file``) can be used
    to find the grid cells that fall inside a buffer around each building
    (150 % of the grid resolution). These indices are computed once and reused
    for the exposure analysis in every generation.

    Parameters
    ----------
    f_bldg_file : str
        Path to the buildings shapefile (needs a ``UID`` field and geometry).
    f_run_path : str
        CityCAT run folder (contains ``citycat.exe``).
    f_idx_rainfall : int
        CityCAT rainfall file number for the indexing run.
    f_idx_config : int
        CityCAT configuration file number for the indexing run.
    f_csv_file : str
        CityCAT maximum-depth CSV written by the indexing run (x, y and depth
        columns after one header row).

    Returns
    -------
    shp_field : str
        Name of the building ID field (``"UID"``).
    cell_index : list of int
        Row number (in the depth CSV) of each grid cell that intersects a
        building buffer.
    buffer_list : list
        Building ID for each entry of ``cell_index``.

    Notes
    -----
    CityCAT is started through the Windows command shell
    (``cd <run_path> & citycat.exe -r R -c C``).
    """
    # Run CityCAT once. Only the grid-cell coordinates of its maximum-depth CSV
    # are used below; the depths themselves are ignored.
    # NOTE (known issue): the command uses the Windows command shell
    # (cd ... & citycat.exe), so it only works on Windows.
    subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                    (run_path = f_run_path, r = f_idx_rainfall, 
                     c = f_idx_config), shell=True)
 
    cols_list = []
    df1 = gpd.read_file(f_bldg_file)  
    for n in df1.columns:
        cols_list.append(n)
    
    shp_field = "UID"  # select the building ID from the shapefile
    # remember to check the buffer value
    buffer_value = 150  # 150 percent of grid or DEM spatial resolution
    
    # Grid resolution dx = spacing of the first two data rows in x (or in y if
    # they share the same x). The buffer is buffer_value % of dx.
    # first get the resolution of the grid
    df_res = pd.read_csv(f_csv_file, nrows=3)
    xdiff = df_res.iloc[2,0] - df_res.iloc[1,0]
    ydiff = df_res.iloc[2,1] - df_res.iloc[1,1]
    if xdiff != 0:
        dx = xdiff
    elif xdiff == 0:
        dx = ydiff
    del(df_res)     
    buffer_distance = ((buffer_value)/100)*dx 
    # buffer distance in percent of grid resolution
    
    # Read the x, y coordinates of every grid cell. A cell's position in these
    # lists (0, 1, 2, ...) is its data-line number in the depth CSV. Depths from
    # later CityCAT runs are matched to cells by this number, which relies on
    # every run listing the cells in the same order.
    x=[]
    y=[]
    with open(f_csv_file, 'r') as t:
        aline = t.readline().strip()  # read header row     
        aline = t.readline()
        # loop through the file and extract x and y values and save them
        while aline != '':
            # split on whitespace, tab or comma: column 0 = x, column 1 = y
            # NOTE (known issue): the pattern is not a raw string (r'...'), so
            # Python 3.12+ shows a SyntaxWarning; the pattern itself works.
            column = re.split('\s|\s\s|\t|,',str(aline))
            # re is the regex library: \s space, \t tab
            x.append(float(column[0]))
            y.append(float(column[1]))
            aline = t.readline()  # read next line   
    t.close()
    
    cell_idx=[]
    for idx, xi in enumerate(x):
        # generate an index based on the line number of the x coordinates 
        cell_idx.append(idx)
    
    # R-tree spatial index of the cell points, so that the cells near each
    # building can be found without checking every cell of the domain.
    index = rtree.index.Index()  # create the spatial index
    for pt_idx, xi, yi in zip(cell_idx,x,y):
        index.insert(pt_idx, (xi,yi))
    del(cell_idx)
    
    cell_index = []  # equal to the line number of the depth file read later
    buffer_list = []
    bldgs = gpd.GeoDataFrame.from_file(f_bldg_file)
    bldgs_df = gpd.GeoDataFrame(bldgs[[str(shp_field), 'geometry']])
    # the columns 'fid' and 'geometry' need to exist as headers
    del(bldgs)
    
    # For each building: draw a buffer polygon around it, take the cells whose
    # points fall inside the buffer's bounding box (fast index query), then
    # keep those that really lie inside the buffer. Each match adds one
    # (building ID, cell line number) pair to buffer_list / cell_index.
    # NOTE (known issue): shapely 2.x warns that the 'resolution' argument of
    # buffer() is deprecated.
    for b_id, b_geom in zip(bldgs_df[str(shp_field)], bldgs_df['geometry']):
        buffer = shape(b_geom.buffer(float(buffer_distance), resolution=10))
        # create a buffer polygon around the building polygon
        for cell in list(index.intersection(buffer.bounds)): 
            # first check if the point is within the bounding box of a
            # building buffer
            cell_int = Point(x[cell], y[cell])  
            if cell_int.intersects(buffer):
                # then check if the point intersects the buffer polygon
                buffer_list.append(b_id)  # store the building ID
                cell_index.append(cell)   # store the line index of the intersecting points
    
    # (df_b is a sorted table of the same pairs; it is not returned or used)
    df_b = pd.DataFrame(list(zip(buffer_list, cell_index)), 
                        columns=[str(shp_field),'cell']) 
    df_b = df_b.sort_values(by=['cell'])    
    print('\n' 'Spatial indices created')
    return shp_field, cell_index, buffer_list  

#-----------------------------------------------------------------------------#
# Read DEM ASCII file 
#-----------------------------------------------------------------------------#
def read_dem_ascii(file_path):
    """Read a DEM stored as an ESRI ASCII grid (``.asc``).

    Parameters
    ----------
    file_path : str
        Path to the ``.asc`` file. The first six lines must be the header
        (``ncols``, ``nrows``, ``xllcorner``, ``yllcorner``, ``cellsize``,
        ``NODATA_value``).

    Returns
    -------
    metadata : dict
        Header keys mapped to their values (as strings).
    data : numpy.ndarray
        Elevations, shape (nrows, ncols).
    """
    with open(file_path, 'r') as file:
        # header: six 'key value' lines (ncols, nrows, xllcorner, yllcorner,
        # cellsize, NODATA_value); the values are kept as strings
        metadata = {}
        for _ in range(6):
            line = file.readline().strip()
            key, value = line.split()
            metadata[key] = value
        
        # the rest of the file is the elevation grid, one row of cells per line
        # (northernmost row first)
        data = np.loadtxt(file)
    
    return metadata, data


#-----------------------------------------------------------------------------#
# Create initial random population
#-----------------------------------------------------------------------------#
def initial_pop(f_pop_size,f_chrom_len_ponds_locat, f_chrom_len_ponds_size,
                                    f_ponds_gene_start, f_ponds_gene_end):
    # generate random population for location chromosomes
    """Create the initial random population of pond configurations.

    Each candidate solution has three linked forms: a location chromosome (one
    binary gene per pond: pond used or not), a size chromosome (a block of
    binary size-step genes for each pond) and the pond size values (number of
    size steps switched on for each pond). Random size chromosomes are drawn
    until they are all different. The second-to-last location chromosome is
    all zeros (no-pond baseline) and the last location and size chromosomes are
    all ones (every pond at full size). Size genes of ponds that are switched
    off are set to zero.

    Parameters
    ----------
    f_pop_size : int
        Number of candidate solutions.
    f_chrom_len_ponds_locat : int
        Number of candidate ponds (length of the location chromosome).
    f_chrom_len_ponds_size : int
        Total number of size steps over all ponds (length of the size
        chromosome).
    f_ponds_gene_start, f_ponds_gene_end : numpy.ndarray
        Start (inclusive) and end (exclusive) of each pond's block of genes in
        the size chromosome.

    Returns
    -------
    i_pop_ponds_locat : numpy.ndarray
        Location chromosomes, shape (f_pop_size, f_chrom_len_ponds_locat).
    i_pop_ponds_size_uniq : numpy.ndarray
        Size chromosomes, shape (f_pop_size, f_chrom_len_ponds_size).
    i_pop_ponds_size_val : numpy.ndarray
        Number of size steps switched on for each pond, shape
        (f_pop_size, f_chrom_len_ponds_locat).
    """
    # location chromosomes: random 0/1, shape (f_pop_size, number of ponds)
    i_pop_ponds_locat = np.random.randint(2, size = (f_pop_size,
                                                    f_chrom_len_ponds_locat))
    
    # set one chromosome to zero (no intervention baseline scenario)
    i_pop_ponds_locat[len(i_pop_ponds_locat)-2] = 0
    # ... and the last one to all ones (every pond switched on)
    i_pop_ponds_locat[len(i_pop_ponds_locat)-1] = 1

    # Size chromosomes: random 0/1, shape (f_pop_size, f_chrom_len_ponds_size),
    # redrawn until all rows are different.
    # generate random population for size chromosomes
    i_pop_ponds_size = np.random.randint(2, size = (f_pop_size,
                                                    f_chrom_len_ponds_size))
    [i_pop_ponds_size_uniq,_] = remove_duplicate_same_population(
                                                            i_pop_ponds_size)
    while len(i_pop_ponds_size_uniq) < f_pop_size:
        del i_pop_ponds_size, i_pop_ponds_size_uniq
        # generate random population
        i_pop_ponds_size = np.random.randint(2, size = (f_pop_size,
                                                    f_chrom_len_ponds_size))
        [i_pop_ponds_size_uniq,_] = remove_duplicate_same_population(
                                                            i_pop_ponds_size)
    
    # last design: every size gene on, so (with every pond switched on) every
    # pond is at full size
    i_pop_ponds_size_uniq[len(i_pop_ponds_size_uniq)-1] = 1
    
    # Switch off the size genes of every pond whose location gene is 0, so that
    # design has no pond there. (Rows were made unique before this step, so two
    # designs can end up with the same size values after it.)
    for i in range(f_pop_size):
        for j in range(f_chrom_len_ponds_locat):
            if i_pop_ponds_locat[i, j] == 0:
                i_pop_ponds_size_uniq[i,
                            f_ponds_gene_start[j]:f_ponds_gene_end[j]] = 0
                
    # generate chromosomes for pond size values
    # size value of pond j = number of its size genes that are 1
    i_pop_ponds_size_val = np.zeros([f_pop_size,f_chrom_len_ponds_locat])
    
    for i in range(f_pop_size):
        for j in range(f_chrom_len_ponds_locat):
            i_pop_ponds_size_val[i,j] = np.sum(
                    i_pop_ponds_size_uniq[i,
                                f_ponds_gene_start[j]:f_ponds_gene_end[j]])
    
    return i_pop_ponds_locat, i_pop_ponds_size_uniq, i_pop_ponds_size_val


#-----------------------------------------------------------------------------#
# Calculate area under the curve (AUC)
#-----------------------------------------------------------------------------#
def auc_func(cost, damage):
    """Area under the cost-damage curve, used for the termination criterion.

    Solutions are sorted by cost, damage is shifted so that the lowest damage
    is zero, and the area is integrated with the trapezoidal rule. The main
    script stops when this area stops decreasing between generations.

    Parameters
    ----------
    cost : array_like
        Intervention cost of each solution, shape (n, 1) or (n,).
    damage : array_like
        Damage of each solution, same shape as ``cost``.

    Returns
    -------
    float
        Area under the curve.
    """
    # Sort the designs by cost, shift damage so that the lowest is 0, then
    # integrate damage over cost with the trapezoidal rule. As the designs
    # improve, the curve moves down and this area shrinks; the main script
    # stops when it no longer shrinks noticeably.
    arr = np.column_stack((cost, damage))
    df = pd.DataFrame(arr, columns=['Cost', 'Damage'])
    df = df.sort_values(by='Cost')
    min_damage = df["Damage"].min()
    damage_adjust = df["Damage"].to_numpy() - min_damage
    # NOTE (known issue): np.trapz was removed in recent NumPy 2.x releases,
    # which is why requirements.txt keeps numpy < 2 (np.trapezoid replaces it).
    auc_var = np.trapz(damage_adjust, df['Cost'])
    return auc_var

#-----------------------------------------------------------------------------#
# Data preparation, CityCAT simulations, and exposure calculation
#-----------------------------------------------------------------------------#
def citycat_exposure(f_bldg_file,
                     f_chrom_len_ponds_locat, f_pop_size, f_run_path,
                     f_rainfall_num, f_config_num, f_pop_ponds_size_val,
                     f_ponds_grid_shp1, f_ponds_depth, f_ponds_grid_cell_area,
                     f_ponds_area_step_val,
                     f_pond_unit_vol_lcc, f_catch_shp,
                     f_dem_metadata, f_dem_data,
                     f_shp_field, f_cell_index,
                     f_buffer_list, f_generation, f_res_dd, f_nonres_dd):
           
    # cost of combined features
    """Simulate each candidate solution with CityCAT and work out its cost and damage.

    For every candidate solution the function:

    1. picks the grid cells of each pond (the cells nearest the pond's lowest
       point, as many as its size steps allow) and works out pond volumes and
       life cycle cost;
    2. burns the ponds into the DEM and writes ``Domain_DEM.asc`` in
       ``f_run_path``;
    3. runs CityCAT once for each rainfall (return period);
    4. reads the maximum water depths around each building, classifies
       buildings as low, medium or high exposure, and calculates direct damage
       from the depth-damage curves (residential per building; retail, offices
       and public buildings per square metre of building area).

    Parameters
    ----------
    f_bldg_file : str
        Path to the buildings shapefile (fields ``UID`` and ``Type``).
    f_chrom_len_ponds_locat : int
        Number of candidate ponds.
    f_pop_size : int
        Number of candidate solutions to simulate.
    f_run_path : str
        CityCAT run folder.
    f_rainfall_num : numpy.ndarray
        CityCAT rainfall file numbers, one per return period.
    f_config_num : int
        CityCAT configuration file number.
    f_pop_ponds_size_val : numpy.ndarray
        Number of size steps for each pond in each solution, shape
        (f_pop_size, f_chrom_len_ponds_locat).
    f_ponds_grid_shp1 : geopandas.GeoDataFrame
        Pond grid cells with the fields prepared in ``configure12_main.py``
        (``Id``, ``elevation``, ``lev_vol``, ``pond_cell_total_depth``).
    f_ponds_depth : numpy.ndarray
        Depth of each pond in metres, shape (1, n_ponds).
    f_ponds_grid_cell_area : float
        Area of one pond grid cell in square metres.
    f_ponds_area_step_val : numpy.ndarray
        Number of grid cells in one size step for each pond, shape (1, n_ponds).
    f_pond_unit_vol_lcc : float
        Pond life cycle cost per cubic metre (GBP).
    f_catch_shp : geopandas.GeoDataFrame
        Polygon covering the DEM extent.
    f_dem_metadata : dict
        DEM header from ``read_dem_ascii``.
    f_dem_data : numpy.ndarray
        Original DEM elevations from ``read_dem_ascii``.
    f_shp_field, f_cell_index, f_buffer_list
        Building ID field and building-to-cell links from ``spatial_indices``.
    f_generation : int
        Current generation (used in progress messages).
    f_res_dd : numpy.ndarray
        Residential depth-damage table: depth (m), damage (GBP).
    f_nonres_dd : numpy.ndarray
        Non-residential depth-damage table: depth (m), then damage per square
        metre (GBP) for retail, offices and public buildings.

    Returns
    -------
    f_chrom_ponds_volume : numpy.ndarray
        Volume of each pond (m3), shape (f_pop_size, n_ponds).
    f_chrom_cost : numpy.ndarray
        Life cycle cost of each solution (million GBP), shape (f_pop_size, 1).
    f_expo_high, f_expo_medium, f_expo_low : numpy.ndarray
        Number of buildings in each exposure class, shape
        (f_pop_size, n_rainfalls).
    f_damage_cost : numpy.ndarray
        Direct damage (GBP) for each rainfall, shape (f_pop_size, n_rainfalls).

    Notes
    -----
    CityCAT is started through the Windows command shell
    (``cd <run_path> & citycat.exe -r R -c C``).
    """
    # Outputs, one row per design:
    #   f_chrom_cost          pond life cycle cost (million £), shape (n, 1)
    #   f_chrom_cost_penalty  (not used)
    #   f_chrom_ponds_volume  volume of each pond (m3), shape (n, number of ponds)
    f_chrom_cost = np.zeros(f_pop_size)[:,np.newaxis]
    f_chrom_cost_penalty = np.zeros(f_pop_size)[:,np.newaxis]
    f_chrom_ponds_volume = np.zeros([f_pop_size, f_chrom_len_ponds_locat])
    # instead of int(f_chrom_len/f_pond_size_steps), we can also use
    # len(f_ponds_grid_shp1['Id'].unique())
    
    # store building exposure
    # (number of buildings per class; shape (n designs, n rainfalls))
    f_expo_low = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    f_expo_medium = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    f_expo_high = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    
    # store damage cost
    # (integer array, so each total damage is stored in whole pounds)
    f_damage_cost = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    # store damage cost in millions
    # (filled in below but not returned)
    f_damage_cost_m = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(float)
    

    # For each design: Part 1 select the pond cells and work out volume and
    # cost; Part 2 burn the ponds into the DEM; Part 3 run CityCAT once per
    # rainfall; Part 4 work out building exposure and direct damage.
    for pop in range(len(f_pop_ponds_size_val)): 
        # this loop iterates for each chromosome
        print("\n" "Gen.{0}: CityCAT simulating chromosome no. {1} of {2}"
              .format(str(f_generation), str(pop+1),
                                          str(len(f_pop_ponds_size_val))))    
        # Remove the DEM written for the previous design.
        # NOTE (known issue): the path is joined with a backslash, so this check
        # only finds the file on Windows; elsewhere the file is simply overwritten
        # further down.
        if os.path.isfile(f_run_path + '\Domain_DEM.asc'):
           os.remove(f_run_path + '\Domain_DEM.asc')

        # ---- Part 1: pond cells, volume and cost ----
        # assign unit depth to genes with value 1
        # f_ponds_area_steps: number of pond grid cells to dig for each pond =
        # size steps x cells per step (4 with the default settings);
        # shape (1, number of ponds)
        f_ponds_area_steps = f_pop_ponds_size_val[pop,:] * f_ponds_area_step_val
        
        # Design with at least one pond: build a modified DEM. (A design without
        # ponds uses the original DEM; see the else branch further down.)
        if np.sum(f_ponds_area_steps) != 0:
            # get indices where f_ponds_depth has a value, not zero
            # index plus one because index starts from zero but data id starts from one
            # f_available_ponds: Ids of the ponds this design builds. Pond Id = column
            # index + 1, so pond Ids must run 1, 2, ..., number of ponds.
            f_available_ponds = [index + 1 for index, value in enumerate(
                                            f_ponds_area_steps[0]) if value != 0]
            
            # ensure that the original file is processed in every iteration
            f_ponds_grid_shp = copy.deepcopy(f_ponds_grid_shp1)
            
            # filter the GeoDataFrame based on these indices
            # here Id is the pond id
            f_ponds_grid_shp_filter = f_ponds_grid_shp[f_ponds_grid_shp
                                                ['Id'].isin(f_available_ponds)]
            
            # compute the minimum elevation for each object ID in 
            # a separate geodataframe
            # (one row per selected pond: its lowest cell)
            f_ponds_min_elevations = f_ponds_grid_shp_filter.loc[
                f_ponds_grid_shp_filter.groupby('Id')['elevation']
                                            .idxmin()].reset_index(drop=True)
            
            f_ponds_area_ids = np.array(f_available_ponds)
            
            f_ponds_area_val = f_ponds_area_steps[0,f_ponds_area_ids-1].astype(int)
            # subtracted one as pond Id starts from one while indices start from zero
            
            # {pond Id: number of cells to dig}
            f_ponds_nearest_count = dict(zip(f_ponds_area_ids, f_ponds_area_val))
            
            # A pond of n cells is made of the n cells closest to the pond's lowest cell
            # (distance 0 for the lowest cell itself), so a pond grows outwards from its
            # lowest point as its size value increases.
            # function to find nearest polygons
            def find_nearest_cells(gdf_subset, min_elevation_row, n):
                min_geom = min_elevation_row.geometry
                gdf_subset = gdf_subset.copy()  # ensure we work with a copy
                gdf_subset.loc[:, 'distance'] = gdf_subset.geometry.apply(
                                            lambda x: min_geom.distance(x))
                nearest = gdf_subset.nsmallest(n, 'distance') 
                return nearest
            
            # DataFrame to store the results
            results = []
            
            # process each object ID separately
            for obj_id in f_ponds_min_elevations['Id']:
                if obj_id in f_ponds_nearest_count:
                    n = f_ponds_nearest_count[obj_id]
                    f_ponds_subset = f_ponds_grid_shp_filter[
                        f_ponds_grid_shp_filter['Id'] == obj_id]
                    min_elevation_row = f_ponds_min_elevations[
                        f_ponds_min_elevations['Id'] == obj_id].iloc[0]
                    nearest_cells = find_nearest_cells(f_ponds_subset,
                                                       min_elevation_row, n)
                    results.append(nearest_cells)
            
            # concatenate all results into a single GeoDataFrame
            f_ponds_final_grid = gpd.GeoDataFrame(pd.concat(results,
                                                            ignore_index=True))
            
            # drop the distance column before saving
            f_ponds_final_grid = f_ponds_final_grid.drop(columns=['distance'])
                       
            # calculate levelling volume of each pond in cubic metres
            f_ponds_level_vol = f_ponds_final_grid.groupby('Id'
                                            )['lev_vol'].sum().reset_index()
    
            # (f_ponds_level_vol_array is not used further: the levelling volume is
            # not part of the cost)
            f_ponds_level_vol_array = np.zeros(f_ponds_area_steps.shape)
            
            f_ponds_level_vol_array[0,f_ponds_area_ids-1] = (
                                                f_ponds_level_vol['lev_vol'])
                                              
             
            # pond volume (m3) = cells dug x cell area (m2) x pond depth (m);
            # shape (1, number of ponds), 0 for ponds that are not built
            f_ponds_volume = (f_ponds_area_steps * f_ponds_grid_cell_area * 
                                                            f_ponds_depth)
            # here f_ponds_area represents the total number of grid cells of a pond
             
            f_chrom_ponds_volume[pop,:] = f_ponds_volume
            
            
            # pond cost (million £) = volume x life cycle cost per m3 / 1,000,000,
            # rounded to 3 decimals; the design's cost objective is the sum over its
            # ponds
            f_ponds_cost = np.round((f_ponds_volume * f_pond_unit_vol_lcc)
                                                                /1000000,3)
            # (not used further)
            f_ponds_location = np.where(f_ponds_cost != 0, 1, 0)
          
            
            # calculate total cost (million £) of all ponds in the chromosome
            f_chrom_cost[pop] =  round(np.sum(f_ponds_cost),3)
            
            # ---- Part 2: burn the ponds into the DEM ----
            # Cut the pond cells out of the catchment polygon and give the rest a depth
            # change of 0; the pond cells keep their pond_cell_total_depth (negative).
            # overlay with the catchment shapefile
            f_ponds_final_grid_catch_subtract = gpd.overlay(f_catch_shp,
                                        f_ponds_final_grid, how='difference')
            
            # assign a new column for pond total depth and set to zero
            f_ponds_final_grid_catch_subtract['pond_cell_total_depth'] = 0
            f_ponds_final_grid_catch_merg = gpd.GeoDataFrame(pd.concat(
                [f_ponds_final_grid, f_ponds_final_grid_catch_subtract], 
                    ignore_index=True), crs = f_ponds_grid_shp_filter.crs)

            
            # Raster grid identical to the original DEM. The header values are
            # converted with int(), so an integer cell size and integer corner
            # coordinates are assumed.
            # create a raster
            cell_size = int(f_dem_metadata['cellsize'])
                    
            columns = int(f_dem_metadata['ncols'])
            rows = int(f_dem_metadata['nrows'])
            
            xmin = int(f_dem_metadata['xllcorner'])
            ymin = int(f_dem_metadata['yllcorner'])
            xmax = xmin + (columns * cell_size)
            ymax = ymin + (rows * cell_size)
            
            # from_origin takes the upper-left corner (xmin, ymax) and the cell size
            f_transform = from_origin(xmin, ymax, cell_size, cell_size)
            
            # rasterize: each DEM cell gets the depth change of the polygon covering
            # its centre (0 where there is none)
            # extract geometry and attribute values
            f_shapes = ((geom, value) for geom, value 
                      in zip(f_ponds_final_grid_catch_merg.geometry,
                    f_ponds_final_grid_catch_merg['pond_cell_total_depth']))
            f_ponds_burned_raster = rasterize(shapes=f_shapes, fill=0,
                                      out_shape=(rows, columns),
                                      transform=f_transform)
     
            # lower the pond cells: new elevation = original + (negative) depth change
            f_ponds_dem = f_dem_data + f_ponds_burned_raster
        
        else:
            # no pond in this design: CityCAT uses the original DEM
            f_ponds_dem = copy.deepcopy(f_dem_data)
            
            
        # Write the DEM that CityCAT will use (Domain_DEM.asc in the run folder):
        # the same header, then the elevations with 2 decimals.
        f_ponds_dem_path =  os.path.join(f_run_path, 'Domain_DEM.asc')
        
        with open(f_ponds_dem_path, 'w') as file:
            for key, value in f_dem_metadata.items():
                file.write(f"{key} {value}\n")
            
            np.savetxt(file, f_ponds_dem, fmt='%.2f')

        
        # ---- Part 3: run CityCAT once for each rainfall (return period) ----
        """ Run CityCAT """
        for rp in range(len(f_rainfall_num)):
            
            print("\n" "Simulating for Rainfall {0}".format(str(f_rainfall_num[rp])))
            
            # CityCAT writes its results to R<rainfall>C<config>_SurfaceMaps. An old
            # folder of that name is deleted first, so stale results are never read.
            run_folder = 'R{r}C{c}_SurfaceMaps'.format(r = f_rainfall_num[rp], 
                                                       c = f_config_num)
            # CityCAT output files path
            citycat_outputs_path = os.path.join(f_run_path, run_folder)
            
            # CityCAT output file name
            citycat_output_file = 'R{r}_C{c}_max_depth.csv'.format(
                                    r = f_rainfall_num[rp], c = f_config_num)
            
            if os.path.exists(citycat_outputs_path):
                shutil.rmtree(citycat_outputs_path)
            
            start_timestamp = pd.Timestamp.now()
            # print("CityCAT run = " + str(run))
            # NOTE (known issue): started through the Windows command shell, so this
            # only works on Windows.
            subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                            (run_path = f_run_path, r = f_rainfall_num[rp],
                             c = f_config_num), shell=True)
            end_timestamp = pd.Timestamp.now()
            run_time = (end_timestamp - start_timestamp)
            print ("Time taken: " + str(run_time))
          
            
            # ---- Part 4: building exposure and direct damage ----
            # (method adapted from the reference in the file header)
            """ Exposure calculation """
    
            # Z: maximum water depth (m) of every grid cell, in file order (column 2
            # of each line after the header)
            f=open(citycat_outputs_path + '/' + citycat_output_file)   
            Z=[]
            aline = f.readline().strip()       
            aline = f.readline()
            while aline != '':
                column = re.split('\s|\s\s|\t|,',str(aline))
                Z.append(float(column[2]))
                aline = f.readline()
            f.close()              
    
            # spatial intersection and classification
            # the line below reads the depth values from the file according to
            # the cell index from above and stores the depth with the intersecting
            # building ID
            df = pd.DataFrame(list(zip(itemgetter(*f_cell_index)(Z),
                                       f_buffer_list)), 
                              columns=['depth',str(f_shp_field)]) 
            del(Z)
    
            # based on the building ID the mean and maximum depths are
            # calculated and stored in a new data frame
            # for each building: mean and 90th-percentile depth over its linked cells
            # (rounded to 3 decimals)
            mean_depth = pd.DataFrame(df.groupby([str(f_shp_field)])
                            ['depth'].mean().astype(float)).round(3).reset_index(
                            level=0).rename(columns={'depth':'mean_depth'})  
            
            p90ile_depth = pd.DataFrame(df.groupby([str(f_shp_field)])
                            ['depth'].quantile(0.90).astype(float)
                            ).round(3).reset_index(level=0).rename(
                                columns={'depth':'p90ile_depth'})
            
            categ_df = pd.merge(mean_depth, p90ile_depth)
            del(mean_depth, p90ile_depth)
            
            # get data from buildings shapefile
            bldgs_data = gpd.read_file(f_bldg_file)
            bldgs_df = gpd.GeoDataFrame(bldgs_data[[str(f_shp_field), 
                                                    'geometry', 'Type']])  
            # calculate the area for each building
            # (m2 for a metric coordinate system, truncated to a whole number)
            bldgs_df['area'] = (bldgs_df.area).astype(int) 
            buildings_join = bldgs_df.merge(categ_df, on=str(f_shp_field), 
                                            how='left')
            
    
            # Exposure classes (depths in m, p90 = 90th-percentile depth):
            #   Low:    mean < 0.10 and p90 < 0.30, or no grid cell linked to it
            #   Medium: mean < 0.10 and p90 >= 0.30, or 0.10 <= mean < 0.30 and p90 < 0.30
            #   High:   mean >= 0.10 and p90 >= 0.30
            # conditions for classifying buildings according 
            # to the threshold values
            low_expo_cond = ((buildings_join['mean_depth'] >= 0) & 
                            (buildings_join['mean_depth'] < 0.10) & 
                            (buildings_join['p90ile_depth'] < 0.30))
                            
            nan_val_cond = ((buildings_join['mean_depth'].isnull()) & 
                            (buildings_join['p90ile_depth'].isnull()))
            
            med_expo_cond1 = ((buildings_join['mean_depth'] >= 0) & 
                             (buildings_join['mean_depth'] < 0.10) & 
                             (buildings_join['p90ile_depth'] >= 0.30))
            
            med_expo_cond2 = ((buildings_join['mean_depth'] >= 0.10) & 
                             (buildings_join['mean_depth'] < 0.30) & 
                             (buildings_join['p90ile_depth'] < 0.30))
            
            high_expo_cond = ((buildings_join['mean_depth'] >= 0.10) & 
                             (buildings_join['p90ile_depth'] >= 0.30))
            
            # condition for selecting different types of buildings
            residential = buildings_join['Type'] == 'Residential'
            retail = buildings_join['Type'] == 'Retail'
            offices = buildings_join['Type'] == 'Offices'
            public_buildings = buildings_join['Type'] == 'Public Buildings'
            
            # add new columns in the geodataframe
            buildings_join['class'] = ''
            # NOTE (known issue): dmg_cost starts as an integer column; pandas 3 refuses
            # to store fractional damage values in it, which is why requirements.txt
            # keeps pandas < 3.
            buildings_join['dmg_cost'] = 0
            
            # classify buildings
            # low exposure
            buildings_join.loc[low_expo_cond, 'class'] = 'A) Low'
             
            # deal with NaN values
            buildings_join.loc[nan_val_cond, 'class'] = 'A) Low'
    
            # medium exposure
            buildings_join.loc[med_expo_cond1, 'class'] = 'B) Medium'
            buildings_join.loc[med_expo_cond2, 'class'] = 'B) Medium'
    
            # high exposure
            buildings_join.loc[high_expo_cond, 'class'] = 'C) High'
            
            # damage cost for low exposure
            buildings_join.loc[low_expo_cond, 'dmg_cost'] = 0  
            buildings_join.loc[nan_val_cond, 'dmg_cost'] = 0
            
            # Damage is calculated for Medium and High buildings only; Low buildings
            # have zero damage.
            # damage cost for medium and high exposure
            # buildings selection condition
            select_cond = (buildings_join['class'] != 'A) Low')
            
            # this would select only high exposure buildings for damage cost
            # calculation if required
            # select_cond = (buildings_join['class'] == 'C) High')
            
            # Residential: damage per building (£), interpolated from the residential
            # table at the building's p90 depth.
            # Retail, offices, public buildings (below): damage per m2 (£), interpolated
            # at the p90 depth from column 1, 2 or 3 of the non-residential table,
            # multiplied by the building area.
            # depth damage cost calculation for residential buildings
            # select 90th percentile depths
            p90_depth = (buildings_join.loc[select_cond & residential, 
                                            'p90ile_depth'])
            # find depth values greater than 3 m and replace with 3
            # MCM does not give damage cost beyond 3 m and we do not extrapolate
            p90_depth[p90_depth > 3] = 3
            
            buildings_join.loc[select_cond & residential, 'dmg_cost'] = (
                np.interp(p90_depth, f_res_dd[:,0], f_res_dd[:,1])) 
                                                
            # depth damage cost calculation for non residential buildings (retail)
            # select 90th percentile depths
            p90_depth = (buildings_join.loc[select_cond & retail, 'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            # for non residential buildings, we also need area
            # see multi colour manual for details
            b_area = (buildings_join.loc[select_cond & retail, 'area'])
            buildings_join.loc[select_cond & retail, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,1])) * b_area)
            
            # depth damage cost calculation for non residential buildings (offices)
            p90_depth = (buildings_join.loc[select_cond & offices, 'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            b_area = (buildings_join.loc[select_cond & offices, 'area'])
            buildings_join.loc[select_cond & offices, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,2])) * b_area)
            
            # depth damage cost calculation for non residential buildings
            # (public buildings)
            p90_depth = (buildings_join.loc[select_cond & public_buildings, 
                                            'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            b_area = (buildings_join.loc[select_cond & public_buildings, 'area'])
            buildings_join.loc[select_cond & public_buildings, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,3]))* b_area)
                            
            del(categ_df)
    
            # calculate total buildings in different exposure classes
            f_expo_low[pop,rp]= (buildings_join['class'] == 'A) Low').sum()
            f_expo_medium[pop,rp] = (buildings_join['class'] == 'B) Medium').sum()        
            f_expo_high[pop,rp]= (buildings_join['class'] == 'C) High').sum()
            
            # calculate total damage cost
            # total direct damage (£) over all buildings, stored as whole pounds
            f_damage_cost[pop,rp] = buildings_join['dmg_cost'].sum()
            # damage cost in millions
            f_damage_cost_m[pop,rp] = np.round(f_damage_cost[pop,rp]/1000000,3)
               
    # The last value is the direct damage (£) for each rainfall;
    # configure12_main.py calls it ..._chrom_damage_rps and turns it into EAD.
    return (f_chrom_ponds_volume, f_chrom_cost,
            f_expo_high, f_expo_medium, f_expo_low, f_damage_cost)

#-----------------------------------------------------------------------------#
# Non dominated sorting function
#-----------------------------------------------------------------------------#
def non_dominated_sorting(population_size,f_chroms_obj_record):
    """Sort solutions into Pareto fronts (fast non-dominated sorting of NSGA-II).

    Both objectives are minimised.

    Parameters
    ----------
    population_size : int
        Number of solutions.
    f_chroms_obj_record : numpy.ndarray
        Objective values, shape (population_size, 2): column 0 is cost and
        column 1 is damage.

    Returns
    -------
    dict
        Front number (0 = non-dominated) mapped to the list of solution indices
        in that front.
    """
    # Fast non-dominated sorting (Deb et al., 2002). Design p dominates q when
    # p is no worse than q in both objectives and better in at least one
    # (both minimised: column 0 = cost, column 1 = damage/EAD).
    #   s[p]      designs that p dominates
    #   n[p]      number of designs that dominate p
    #   front[i]  designs in front i (front 0 = dominated by none)
    #   rank[p]   front number of p (filled in but not returned)
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):
            
            # the three conditions together mean: p <= q in both objectives and
            # p < q in at least one, i.e. p dominates q
            if ((f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<=f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                f_chroms_obj_record[p][1]<=f_chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            # q dominates p
            elif ((f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>=f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>=f_chroms_obj_record[q][1])):
                n[p]=n[p]+1
        # nobody dominates p: it belongs to the first front
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    # Peel off the fronts one by one: taking front i away lowers the count n[q]
    # of every design it dominates; those that reach 0 form front i + 1.
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                 
    # the loop ends with an empty front, which is removed
    del front[len(front)-1]
    return front

#-----------------------------------------------------------------------------#
# Calculate crowding distance
#-----------------------------------------------------------------------------#
def calculate_crowding_distance(f_front,f_chroms_obj_record):
    """Crowding distance of each solution within its front (NSGA-II).

    For each objective the solutions of a front are sorted and each one gets
    the normalised distance between its two neighbours; the values for the two
    objectives are added. The solutions at either end of a front get large
    fixed values (666666666 for cost, 333333333 for damage) so that they are
    always kept.

    Parameters
    ----------
    f_front : dict
        Fronts from ``non_dominated_sorting``.
    f_chroms_obj_record : numpy.ndarray
        Objective values, shape (n, 2).

    Returns
    -------
    dict
        Front number mapped to a dict of {solution index: crowding distance}.
    """
    # For each front: sort its designs by cost (dt) and by damage (de). For one
    # objective, an interior design's distance is the gap between its two
    # neighbours divided by the range of that objective in the front; the two
    # end designs get a very large value (666666666 for cost, 333333333 for
    # damage) so they are always preferred. The total is the sum over both
    # objectives: larger = more isolated = preferred, which keeps the front
    # spread out.
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    
    for o in range(len(f_front)):
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_obj_record[k][0]
                de[k] = f_chroms_obj_record[k][1]
            del k
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
    
            # now de_sort and dt_sort keys are not the same, we calculate
            # distance for elements with the same key in both
            key_lst = list(dt_sort.keys())    
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(dt_sort)-1:
                    dt_dis[key] = ((abs(dt_sort[key_lst[i+1]]-
                                        dt_sort[key_lst[i-1]]))/
                                   (dt_sort[key_lst[len(key_lst)-1]]-
                                    dt_sort[key_lst[0]]))
                else:
                    dt_dis[key] = 666666666
            del i,key, key_lst
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            
            t_dis = {}
            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]
            
            distance[o] = t_dis
    
    return distance

#-----------------------------------------------------------------------------#
# Sort population based on rank and crowding distance
#-----------------------------------------------------------------------------#
def fitness_sort(f_distance, f_pop_size):
    """Order solutions by front and then by crowding distance (largest first).

    Parameters
    ----------
    f_distance : dict
        Crowding distances from ``calculate_crowding_distance``.
    f_pop_size : int
        Total number of solutions in ``f_distance``.

    Returns
    -------
    list of int
        Solution indices from fittest to least fit.
    """
    # sort each front by crowding distance, largest first
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    # then list the fronts in order (front 0 first): row numbers from fittest to
    # least fit
    parents_offspring = [None]*f_pop_size
    a = 0
    for i in range(len(f_distance_sort)):
        for j in f_distance_sort[i].keys():
            parents_offspring[a] = j
            a = a+1
    return parents_offspring

#-----------------------------------------------------------------------------#
# Parent selection using binary tournament
#-----------------------------------------------------------------------------#
def fitter_parent(f_sorted_fitness,f_pop_size):
    """Pick a parent by binary tournament.

    Two different solutions are drawn at random and the one ranked higher in
    ``f_sorted_fitness`` is returned.

    Parameters
    ----------
    f_sorted_fitness : numpy.ndarray
        Solution indices from fittest to least fit (from ``fitness_sort``).
    f_pop_size : int
        Population size.

    Returns
    -------
    int
        Index of the selected parent.
    """
    # draw two different designs at random
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    # np.where gives each design's position in the fitness order; the one
    # nearer the front (smaller position) wins
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]
        
#-----------------------------------------------------------------------------#
# Crossover and mutation
#-----------------------------------------------------------------------------#
def crossover_uniform_mutation_bitflip(f_p1, f_p2, f_p1_loc, f_p2_loc,
                    f_co_prob, f_m_prob, f_chrom_len_ponds_locat,
                    f_chrom_len_ponds_size, p_start_locs, p_end_locs):
    
    """Create two children with uniform crossover and bit-flip mutation.

    Crossover and mutation are applied to the location chromosomes and then to
    the size chromosomes. Size genes of ponds that are switched off in a
    child's location chromosome are set to zero. If both children end up with
    the same pond sizes, the second child's size chromosome is drawn again by
    crossover (up to 5000 attempts).

    Parameters
    ----------
    f_p1, f_p2 : numpy.ndarray
        Size chromosomes of the two parents.
    f_p1_loc, f_p2_loc : numpy.ndarray
        Location chromosomes of the two parents.
    f_co_prob : float
        Probability of taking each gene from the other parent.
    f_m_prob : array_like
        Mutation probabilities: [location genes, size genes].
    f_chrom_len_ponds_locat : int
        Number of candidate ponds.
    f_chrom_len_ponds_size : int
        Length of the size chromosome.
    p_start_locs, p_end_locs : numpy.ndarray
        Start (inclusive) and end (exclusive) of each pond's genes in the size
        chromosome.

    Returns
    -------
    f_c1, f_c2 : numpy.ndarray
        Size chromosomes of the two children.
    f_c1_ponds_value, f_c2_ponds_value : numpy.ndarray
        Number of size steps switched on for each pond in each child.
    """
    # the children start as copies of the parents
    f_c1, f_c2 = copy.deepcopy(f_p1), copy.deepcopy(f_p2)
    f_c1_loc, f_c2_loc = copy.deepcopy(f_p1_loc), copy.deepcopy(f_p2_loc)
    
    # Uniform crossover: each gene of child 1 is replaced by parent 2's gene
    # with probability f_co_prob (and each gene of child 2 by parent 1's).
    # The location and size chromosomes are crossed separately.
    # create children for location chromosomes
    for i in range(len(f_c1_loc)):
        if np.random.rand() < f_co_prob:
            f_c1_loc[i] = f_p2_loc[i]

    for i in range(len(f_c2_loc)):
        if np.random.rand() < f_co_prob:
            f_c2_loc[i] = f_p1_loc[i]
    
    # Bit-flip mutation: each gene flips 0 <-> 1 with probability f_m_prob[0]
    # (location genes) or f_m_prob[1] (size genes, below).
    # mutate location chromosomes
    # first child
    for i in range(len(f_c1_loc)):
        if np.random.rand() < f_m_prob[0]:
            if f_c1_loc[i] == 0:
                f_c1_loc[i] = 1
            else:
                f_c1_loc[i] = 0
    # second child
    for i in range(len(f_c2_loc)):
        if np.random.rand() < f_m_prob[0]:
            if f_c2_loc[i] == 0:
                f_c2_loc[i] = 1
            else:
                f_c2_loc[i] = 0

    # create children for size chromosomes
    for i in range(len(f_c1)):
        if np.random.rand() < f_co_prob:
            f_c1[i] = f_p2[i]

    for i in range(len(f_c2)):
        if np.random.rand() < f_co_prob:
            f_c2[i] = f_p1[i]
    
    # mutate size chromosomes
    # first child
    for i in range(len(f_c1)):
        if np.random.rand() < f_m_prob[1]:
            if f_c1[i] == 0:
                f_c1[i] = 1
            else:
                f_c1[i] = 0
    # second child
    for i in range(len(f_c2)):
        if np.random.rand() < f_m_prob[1]:
            if f_c2[i] == 0:
                f_c2[i] = 1
            else:
                f_c2[i] = 0
    
    
    # A pond switched off in a child's location chromosome gets all its size
    # genes set to 0 (no pond).
    for j in range(f_chrom_len_ponds_locat):
        if f_c1_loc[j] == 0:
            f_c1[p_start_locs[j]:p_end_locs[j]] = 0
        if f_c2_loc[j] == 0:
            f_c2[p_start_locs[j]:p_end_locs[j]] = 0
        
    # check if both children represent the same pond settings
    # size value = number of size genes switched on for each pond
    f_c1_ponds_value = np.zeros(f_chrom_len_ponds_locat)
    f_c2_ponds_value = np.zeros(f_chrom_len_ponds_locat)
    
    for j in range(f_chrom_len_ponds_locat):
        f_c1_ponds_value[j] = np.sum(
                f_c1[p_start_locs[j]:p_end_locs[j]])
        f_c2_ponds_value[j] = np.sum(
                f_c2[p_start_locs[j]:p_end_locs[j]])
    # If both children have identical pond sizes, redraw child 2's size
    # chromosome from the parents (crossover only, no mutation) until it
    # differs, at most 5000 times.
    d = 0
    while np.all(f_c1_ponds_value == f_c2_ponds_value) == True and d < 5000:
        f_c2 = copy.deepcopy(f_p2)
        for i in range(len(f_c2)):
            if np.random.rand() < f_co_prob:
                f_c2[i] = f_p1[i]
        
        for j in range(f_chrom_len_ponds_locat):
            if f_c2_loc[j] == 0:
                f_c2[p_start_locs[j]:p_end_locs[j]] = 0
                
        for j in range(f_chrom_len_ponds_locat):
            f_c2_ponds_value[j] = np.sum(
                    f_c2[p_start_locs[j]:p_end_locs[j]])
        d += 1

    return f_c1, f_c2, f_c1_ponds_value, f_c2_ponds_value


#-----------------------------------------------------------------------------#
# Remove duplicates from a list
#-----------------------------------------------------------------------------#
def remove_duplicate_list(record_list):
    # print('\n' 'Checking duplicates in the list:')
    """Remove repeated rows, keeping the first occurrence.

    Parameters
    ----------
    record_list : numpy.ndarray
        Rows to check, e.g. the objective values of a population.

    Returns
    -------
    m_pool : numpy.ndarray
        ``record_list`` without repeated rows.
    list of int
        Indices of the rows that were removed.
    """
    # Compare every pair of rows; a row equal to an earlier row is marked for
    # removal (dict keys stop a row being marked twice).
    m_pool = copy.deepcopy(record_list)
    idx = {}
    for i in range(0,len(m_pool)):
        for j in range(i+1,len(m_pool)):
            if np.all((m_pool[i] == m_pool[j]) == True):
                # print('Record no. {0} was equal to record no. {1}'
                #       .format(i,j))
                idx[j] = j 
    del i, j
    
    if idx!={}:
        m_pool = np.delete(m_pool, list(idx.values()),0)

    return m_pool, list(idx.values())

#-----------------------------------------------------------------------------#
# Remove duplicates from a population
#-----------------------------------------------------------------------------#
def remove_duplicate_same_population(same_population):
    # print('\nChecking duplicate chroms in the same population:')
    """Remove repeated chromosomes from a population, keeping the first occurrence.

    Parameters
    ----------
    same_population : numpy.ndarray
        Population, one chromosome per row.

    Returns
    -------
    pop_uniq : numpy.ndarray
        Population without repeated chromosomes.
    list of int
        Indices of the chromosomes that were removed.
    """
    # same approach as remove_duplicate_list(): mark rows equal to an earlier
    # row, then delete them
    pop_uniq = copy.deepcopy(same_population)
    a = {}
    for i in range(0, len(pop_uniq)):
        for j in range(i+1,len(pop_uniq)):
            if np.all((pop_uniq[i] == pop_uniq[j]) == True):
                # print('Chrom no. {0}  was equal to chrom no {1}'.format(i,j))
                a[j] = j
    if a!={}:
        pop_uniq = np.delete(pop_uniq, list(a.values()),0)
        # print('....\n {0} duplicate chroms deleted'.format(len(a)))
    return pop_uniq, list(a.values())

#-----------------------------------------------------------------------------#
# Remove duplicate offspring pond settings
#-----------------------------------------------------------------------------#
def remove_duplicate_ponds_settigs(f_offspring_size_val, f_offspring,
                                   f_simulated_population_size_val,
                                   f_simulated_population):
    
    """Drop offspring whose pond sizes have already been simulated.

    Parameters
    ----------
    f_offspring_size_val : numpy.ndarray
        Pond size values (size steps per pond) of the offspring.
    f_offspring : numpy.ndarray
        Size chromosomes of the offspring (same rows as above).
    f_simulated_population_size_val : numpy.ndarray
        Pond size values of every solution simulated so far.
    f_simulated_population : numpy.ndarray
        Size chromosomes of every solution simulated so far (not used in the
        comparison).

    Returns
    -------
    f_offspring_size_val, f_offspring : numpy.ndarray
        The offspring that have not been simulated before.
    """
    f_offspring_size_val = copy.deepcopy(f_offspring_size_val)
    f_offspring = copy.deepcopy(f_offspring)
    
    # Compare each offspring's size values with every design simulated so far;
    # matches are removed so that no design is simulated twice.
    a = {}
    for i, f_sim in enumerate(f_simulated_population_size_val):
        for j , f_offsp in enumerate(f_offspring_size_val):
             if np.all((f_sim == f_offsp) == True):
                a[j] = j
    if a!={}:
        f_offspring_size_val = np.delete(f_offspring_size_val, 
                                                     list(a.values()),0)
        f_offspring = np.delete(f_offspring, list(a.values()),0)

    return f_offspring_size_val, f_offspring
    

#-----------------------------------------------------------------------------#
# Create a single series scatter plot
#-----------------------------------------------------------------------------#
def scatter_plot(f_fig_background, f_plot_title, f_cost, f_exposure,
                 f_plot_legend_series,
                    f_plot_x_limit, f_plot_y_limit, f_plot_x_axis_label,
                    f_plot_y_axis_label, f_save_file):
    
    """Save a scatter plot of cost against damage for one population.

    Parameters
    ----------
    f_fig_background : str
        Matplotlib style, e.g. ``'default'`` or ``'dark_background'``.
    f_plot_title : str
        Plot title.
    f_cost, f_exposure : array_like
        Values for the x axis (cost) and y axis (damage).
    f_plot_legend_series : str
        Legend label.
    f_plot_x_limit, f_plot_y_limit : list of float
        Axis limits as [min, max].
    f_plot_x_axis_label, f_plot_y_axis_label : str
        Axis labels.
    f_save_file : str
        Output file name. Without an extension the figure is saved as PNG
        (12 x 8 inches at 300 dpi).
    """
    plt.style.use(f_fig_background)
    plt.figure(figsize=(12, 8), dpi=300)
    
    ax = plt.subplot()
    
    # set x and y limits
    plt.xlim(f_plot_x_limit[0], f_plot_x_limit[1])
    plt.ylim(f_plot_y_limit[0], f_plot_y_limit[1])
    
    # (both branches use the same edge colour)
    if f_fig_background == 'dark_background':
        edgecolors_evolving = '#2E75B6'
        
    else:
        edgecolors_evolving = '#2E75B6'
    
    # create the scatter plot for the first series
    ax.scatter(f_cost, f_exposure, s= 80, facecolors='#9BC2E6', 
               edgecolors=edgecolors_evolving, linewidth=1.5, 
               alpha=1, marker='o', 
               label='Evolving population')
    
    # the legend text comes from f_plot_legend_series
    # add the series legend
    plt.legend([f_plot_legend_series], 
               loc ="upper right", 
               prop={'weight': 'normal', "size": 14, 
                                           'stretch': 'normal'})
    # format ticks
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tick_params(direction='out', length=6, width=1)
    
    # format grid
    plt.grid(color = '#A6A6A6', linestyle = '-', linewidth = 0.25)
     
    # add labels and title
    plt.xlabel(f_plot_x_axis_label, fontsize = 14)
    plt.ylabel(f_plot_y_axis_label, fontsize = 14)
       
    plt.title(f_plot_title, fontsize = 18)
    
    # f_save_file has no extension, so matplotlib saves a PNG
    # save figure
    plt.savefig(f_save_file, dpi='figure', transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.5)
    plt.close()
    

#-----------------------------------------------------------------------------#
# Separate new and old individuals
#-----------------------------------------------------------------------------#
def separate_new_old(f_new_population, f_old_population):
    # print('\n' 'Checking new and old chroms in new population')
    """Split a new population into solutions kept from the old one and new ones.

    Parameters
    ----------
    f_new_population, f_old_population : numpy.ndarray
        Populations to compare, one solution per row.

    Returns
    -------
    f_new_chroms : numpy.ndarray
        Rows of the new population that are not in the old population.
    f_old_chroms : numpy.ndarray
        Rows of the new population that were already in the old population.
    f_old_chroms_index : list of int
        Row indices (in the new population) of ``f_old_chroms``.

    Notes
    -----
    If no solution of the old population is kept, ``f_old_chroms_index`` is
    never assigned and the function raises ``UnboundLocalError``.
    """
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    # for new and old we used f_new_population. For new_chroms, we
    # delete old chroms from the new population. For old_chroms, we
    # select old chroms from the new population
    # mark every row of the new population that also appears in the old one
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                # print('Chromosome no. {0} in new population was '.format(i) +  
                #       'equal to chromosome no. {0} in old population'
                #       .format(j))
                a[j] = j
    # NOTE (known issue): f_old_chroms_index is only assigned inside this if
    # block, so the return below raises UnboundLocalError when no design of
    # the old population was kept.
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]
        f_old_chroms_index = list(a.values())
    return f_new_chroms, f_old_chroms, f_old_chroms_index


#-----------------------------------------------------------------------------#
# Delete chromosomes that have the same objective functions
#-----------------------------------------------------------------------------#
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    # print('\n' 'Checking duplicate chroms in different populations:')
    """Delete the given rows from a population.

    Used to drop solutions whose objective values repeat those of another
    solution (row indices from ``remove_duplicate_list``).

    Parameters
    ----------
    f_comb_population : numpy.ndarray
        Population, one solution per row.
    f_dup_idx_obj : list of int
        Row indices to delete.

    Returns
    -------
    numpy.ndarray
        Population without those rows.
    """
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)

    # Delete the listed rows (indices from remove_duplicate_list()) so that
    # this array stays aligned with the de-duplicated objective values.
    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
        # print('....\n {0} duplicate chroms from population 1 deleted'
        #       .format(len(a)))
    return comb_pop

###############################################################################
""" Functions section ends here """

###############################################################################

