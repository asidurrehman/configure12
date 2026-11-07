# -*- coding: utf-8 -*-
"""
╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝

File name: configure12_main.py

@author: Asid Ur Rehman

Exposure analysis adapted from Robert Berstch
(https://doi.org/10.1016/j.envsoft.2022.105490)

Organisation: Newcastle University

About
------
This script contains the main body of code that runs
CONFIGURE (Cost benefit optimisation Framework for Implementing blue Green
infrastructure), a framework that integrates an evolutionary genetic algorithm
(NSGA II) with CityCAT to find optimal locations and sizes of BGI for
cost effective deployment. This version is designed specifically to optimise
the location and size of detention ponds for multiple rainfall return periods
by calculating Expected Annual Damage (EAD) using Direct Damage Cost (DDC)
for each return period.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import warnings
import copy
import argparse
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

#=============================================================================#
# HOW THIS SCRIPT WORKS
#=============================================================================#
# CONFIGURE v1.2 searches for the best locations and sizes of detention ponds.
# Every candidate design is tested by digging its ponds into the terrain (DEM),
# running the CityCAT flood model for five rainstorms of different return
# periods and calculating the flood damage to buildings. The NSGA-II genetic
# algorithm then evolves the designs towards the best trade-off between two
# objectives, both minimised:
#   - cost: life cycle cost of the ponds (million £)
#   - EAD:  Expected Annual Damage to buildings (million £)
#
# The script runs from top to bottom in these steps (search for "STEP n"):
#   STEP 1  Settings and paths: read config.yaml and move into the run folder.
#   STEP 2  Input data: buildings and depth-damage tables.
#   STEP 3  Building-grid links: one CityCAT run to find the grid cells
#           around each building.
#   STEP 4  Candidate ponds: read the pond grid and the DEM and work out how a
#           design is stored as genes (see NAMING CONVENTIONS below).
#   STEP 5  Optimisation settings: population size, generations, stopping rule.
#   STEP 6  Initial population: create random designs, simulate them and
#           calculate their cost and EAD.
#   STEP 7  Record keeping: arrays that collect results; output column labels.
#   STEP 8  Generation loop, repeated up to max_generations times:
#           8a rank the parents (non-dominated sorting + crowding distance)
#           8b create offspring (tournament selection, uniform crossover,
#              bit-flip mutation), skipping designs already simulated
#           8c simulate the offspring and calculate their cost and EAD
#           8d save every simulated design to simulated_data.csv
#           8e combine parents and offspring; keep the best pop_size designs
#           8f make them the new parents; save generation_data.csv
#           8g stop early if the trade-off curve has stopped improving
#   STEP 9  Final result: save the optimal designs (first Pareto front) to
#           optimised_data.csv and plot them.
#
# Run time is dominated by CityCAT: every design needs one CityCAT run per
# rainfall in rainfall_num (five with the default settings).
#
#-----------------------------------------------------------------------------#
# NAMING CONVENTIONS
#-----------------------------------------------------------------------------#
# Chromosomes: every pond design is stored in three linked forms.
#   ..._locat     location chromosome: one 0/1 gene per candidate pond
#                 (1 = the pond may be built). Length chrom_len_ponds_locat.
#   ..._size      size chromosome: one 0/1 gene per 100 m2 size step of every
#                 pond. Pond j owns the columns ponds_gene_start[j] to
#                 ponds_gene_end[j] - 1. Length chrom_len_ponds_size.
#   ..._size_val  size values: the number of size steps switched on for each
#                 pond (sum of that pond's size genes; 0 = no pond). This is
#                 the form that is simulated and compared, and it is exported
#                 as the Size_Steps_Pond_<j> columns.
#   A pond is only built if its size value is above zero.
#   (pop_size is the population size, not a chromosome.)
#
# Prefixes of population arrays (one row per design):
#   i_           initial (random) population
#   p_           parents: the current population, replaced every generation
#   o_           offspring created in the current generation
#   comb_        parents and offspring combined
#   f_           the fittest designs chosen from comb_ to become the next
#                parents (in configure12_functions.py f_ marks function
#                arguments instead)
#   gen_         every generation's population, stacked (generation_data.csv)
#   simulated_   every design simulated so far (simulated_data.csv)
#   opt_         the final optimal designs (optimised_data.csv)
# Suffixes and objective arrays:
#   _rps                one column per rainfall, in the order of rainfall_num
#   _uniq_obj           after removing designs whose (cost, EAD) pair repeats
#                       an earlier design's
#   _chrom_cost         cost of each design (million £), shape (n, 1)
#   _chrom_damage       EAD of each design (million £), shape (n, 1)
#   _chrom_damage_rps   direct damage per rainfall (£), shape (n, rainfalls)
#   _chrom_ponds_volume volume of each pond (m3), shape (n, number of ponds)
#   _expo_high/_medium  number of buildings in the high/medium exposure class
#                       for each rainfall, shape (n, rainfalls)
#
#=============================================================================#
# STEP 1: SETTINGS AND PATHS
#=============================================================================#
#-----------------------------------------------------------------------------#
# Read the run configuration (config.yaml)
#-----------------------------------------------------------------------------#
# USER INPUT: run settings live in config.yaml (run_path, rainfall_num,
# config_num, pop_size, max_generations, pond_unit_vol_lcc and the plot axis
# limits). Edit that file rather than this script. To use another file:
#     python configure12_main.py --config my_config.yaml
# SCRIPT_DIR is the folder of this script, used to find config.yaml and the
# functions module.
SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    description="Run CONFIGURE v1.2 (detention pond optimisation with CityCAT)")
parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"),
                    help="Path to the YAML configuration file")
args, _ = parser.parse_known_args()  # parse_known_args keeps Spyder/IPython working
with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load the CONFIGURE functions from the same folder as this script
sys.path.insert(0, str(SCRIPT_DIR))
import configure12_functions as cf

#-----------------------------------------------------------------------------#
# Creating and setting folders and paths
#-----------------------------------------------------------------------------#
# CityCAT run path
run_path = os.path.normpath(config["run_path"])

# run_path must contain citycat.exe, the CityCAT rainfall and configuration
# files and the input data read below (see the README). The script works
# inside this folder, so all output files are written here too.
# Change working directory
os.chdir(run_path)
# Show working directory
print("Current working directory: {0}".format(os.getcwd()))
# CityCAT rainfall file numbers
# Every design is simulated once with each of these CityCAT rainfall files.
# The EAD calculation in STEP 6 expects five of them, ordered from the
# 10-year to the 100-year return period (T = 10, 20, 30, 50, 100 years).
rainfall_num = np.array(config["rainfall_num"])
# CityCAT configuration file number
# (used for all optimisation runs; the one-off run in STEP 3 uses idx_config)
config_num = config["config_num"]

#=============================================================================#
# STEP 2: INPUT DATA
#=============================================================================#
# Buildings shapefile
# Needs a UID field (building ID) and a Type field. Damage is calculated for
# the types Residential, Retail, Offices and Public Buildings; buildings of
# any other type are counted in the exposure classes but get no damage.
bldg_file = os.path.join(run_path, 'vector', 'buildings.shp')

# Settings for the one-off indexing run in STEP 3: CityCAT is run with the
# first rainfall and configuration idx_config, and only the x, y coordinates
# of the grid cells in its maximum-depth CSV are used.
# USER INPUT: idx_config is the CityCAT configuration number for that run.
# CSV file for indexing
idx_rainfall = rainfall_num[0]
idx_config = 1
idx_folder = 'R{r}C{c}_SurfaceMaps'.format(r=idx_rainfall, c=idx_config)
idx_file = 'R{r}_C{c}_max_depth.csv'.format(r=idx_rainfall, c=idx_config)
csv_file = os.path.join(run_path, idx_folder, idx_file)

##------------- Depth Damage Cost calculations ----------------------------##
# Get depth damage cost data from a CSV
# Depth is in metres and cost is in £
# For residential
# residential table: column 0 = depth (m), column 1 = damage per building (£)
df_res_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'residential_damage_calculation.csv')))
# Dataframe to numpy array
res_dd = np.array(df_res_dd_data)

# For nonresidential
# non-residential table: column 0 = depth (m), columns 1-3 = damage per m2 of
# building area (£) for Retail, Offices and Public Buildings.
# Depths above 3 m are capped at 3 m when the tables are used.
df_nonres_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'non_residential_damage_calculation.csv')))
# Dataframe to numpy array
nonres_dd = np.array(df_nonres_dd_data)

#-----------------------------------------------------------------------------#
""" Main code starts here """
#-----------------------------------------------------------------------------#
#=============================================================================#
# STEP 3: LINK BUILDINGS TO GRID CELLS
#=============================================================================#
# spatial_indices() runs CityCAT once and links every building to the grid
# cells that lie within a buffer around it (1.5 x the grid resolution):
#   shp_field    name of the building ID field ("UID")
#   cell_index   line number (in CityCAT's depth CSV) of each linked cell
#   buffer_list  building ID for each entry of cell_index
# These links are reused for every simulation, so the water depths around
# each building can be read quickly from every CityCAT result.
# Get variables needed to calculate building exposure in each
# GA generation
[shp_field, cell_index, buffer_list] = cf.spatial_indices(bldg_file, run_path,
                                       idx_rainfall, idx_config, csv_file)

#=============================================================================#
# STEP 4: CANDIDATE PONDS, DEM AND GENE ENCODING
#=============================================================================#
# all_possible_ponds_grid.shp divides every candidate pond into small square
# grid cells (ponds_grid_cell_area = 25 m2 each). Fields used: Id (pond
# number, expected to run 1, 2, ..., number of ponds) and elevation (ground
# level of the cell). A design decides how many cells of each pond to dig.
# Read ponds shapefile
ponds_grid_shp = gpd.read_file(os.path.join(run_path,
                                    'vector', 'all_possible_ponds_grid.shp'))
# Life cycle cost of a pond (25 years)
# Life cycle cost is calculated as pounds per cubic metre
pond_unit_vol_lcc = config["pond_unit_vol_lcc"]  # 153.2 in the published study
# One time construction cost
pond_unit_vol_copex = 89.42
# Operations and maintenance cost uses inflated values

# Note: pond_unit_vol_copex is not used in any calculation; only the life
# cycle cost (pond_unit_vol_lcc) enters the cost objective.
#
# To ensure the DEM created after modification is not displaced,
# convert the DEM extent to a shapefile
catch_shp = gpd.read_file(os.path.join(run_path, 'vector', 'catchment_dem.shp'))

# catchment_dem.shp (above) is one polygon covering the whole DEM. In
# citycat_exposure() the pond cells are cut out of it and both are rasterised
# together, so the pond raster always covers the full DEM grid.
#
# The original DEM (ESRI ASCII grid). Every design starts from this
# unmodified DEM; the modified copy that CityCAT uses is written to
# <run_path>/Domain_DEM.asc for each design.
dem_original_path = os.path.join(run_path, 'domain_dem_original',
                                 'Domain_DEM.asc')

# Read DEM ASCII file
# dem_metadata: header values (ncols, nrows, xllcorner, ...); dem_data:
# elevations, shape (nrows, ncols)
dem_metadata, dem_data = cf.read_dem_ascii(dem_original_path)

# Count number of ponds
ponds_count = ponds_grid_shp['Id'].nunique()

# Prepare every pond cell for digging. A pond gets a flat base 1 m below the
# lowest ground level of that pond:
#   min_elv    lowest elevation among the cells of the same pond
#   elv_diff   min_elv - elevation (zero or negative): how far the cell must
#              be lowered to reach the pond's lowest point (levelling)
#   pond_cell_total_depth = elv_diff + pond_depth: the total change added to
#              the DEM when the cell is part of a pond (negative = dig down)
# Compute the minimum elevation for each object (pond) ID
min_elv = ponds_grid_shp.groupby('Id')['elevation'].min().reset_index()

# Rename the column in the geodataframe
min_elv.rename(columns={'elevation': 'min_elv'}, inplace=True)

# Merge the minimum elevation back into the original dataframe
ponds_grid_shp = ponds_grid_shp.merge(min_elv, on='Id')

# Calculate elevation removal needed to level the pond base
ponds_grid_shp["elv_diff"] = (ponds_grid_shp["min_elv"] -
                              ponds_grid_shp["elevation"])

# USER INPUT: depth dug below the levelled base (m, negative). If you change
# it, change ponds_stat['pond_depth'] below as well (positive), which is
# used for the pond volume and cost.
ponds_grid_shp["pond_depth"] = -1
# Negative value because this will be subtracted from the DEM

ponds_grid_shp["pond_cell_total_depth"] = (ponds_grid_shp["elv_diff"] +
                          ponds_grid_shp["pond_depth"])

# Calculate area
# (m2, from the cell geometry; assumes a metric coordinate system)
ponds_grid_shp['area'] = ponds_grid_shp.geometry.area

# Volume to remove for pond base levelling
# (m3; summed per pond in citycat_exposure() but not added to the cost)
ponds_grid_shp["lev_vol"] = (abs(ponds_grid_shp["elv_diff"]) *
                                                     ponds_grid_shp['area'])

# Calculate area of each pond in square metres
# ponds_stat: one row per pond, ordered by Id: total area, depth and how the
# pond's size is split into genes.
ponds_stat = ponds_grid_shp.groupby('Id')['area'].sum().reset_index()
ponds_area = np.array(ponds_stat['area'])

# Depth of ponds in metres. The user can assign different depths to different ponds
# USER INPUT: keep this consistent with ponds_grid_shp['pond_depth'] above,
# which sets the depth actually dug into the DEM.
ponds_stat['pond_depth'] = 1
# shape (1, number of ponds)
ponds_depth = np.array(ponds_stat['pond_depth'])[np.newaxis, :]

# Maximum volume in cubic metres
ponds_stat['max_vol'] = ponds_stat['area'] * ponds_stat['pond_depth']

# Pond sizes change in steps of min_area (100 m2). A pond of area A therefore
# has A // 100 size steps, and each step is one gene of the size chromosome.
# USER INPUT: min_area sets the size step (m2).
# Minimum pond area and related volume to consider
ponds_stat['min_area'] = 100
ponds_stat['min_volume'] = ponds_stat['min_area'] * ponds_stat['pond_depth']

# Area steps for each pond
ponds_stat['area_steps'] = (ponds_stat['area'] /
                             ponds_stat['min_area']).astype(int)

# Gene positions: pond j owns the size-chromosome columns gene_start[j] to
# gene_end[j] - 1. Example: area_steps = [3, 2] gives pond 0 the genes 0-2
# and pond 1 the genes 3-4.
# Define starting positions of genes for each pond
ponds_stat['gene_end'] = ponds_stat['area_steps'].cumsum()
ponds_stat['gene_start'] = ponds_stat['gene_end'] - ponds_stat['area_steps']

ponds_gene_start = np.array(ponds_stat['gene_start'])
ponds_gene_end = np.array(ponds_stat['gene_end'])

# Depth step value for each pond
# (More precisely, this block works out how many pond grid cells make up one
# size step: area_step_val = min_area / ponds_grid_cell_area = 100 / 25 = 4.)
# USER INPUT: ponds_grid_cell_area must equal the area (m2) of one cell of
# all_possible_ponds_grid.shp.
ponds_grid_cell_area = 25
ponds_stat['grid_cell_area'] = ponds_grid_cell_area
ponds_stat['area_step_val'] = ponds_stat['min_area'] / ponds_stat['grid_cell_area']
# 25 is the pond grid cell area
ponds_area_step_val = np.array(ponds_stat['area_step_val'])[np.newaxis, :]

# Chromosome lengths: one location gene per pond; one size gene per size step
# of every pond.
# Chromosome length for location optimisation
chrom_len_ponds_locat = ponds_count

# Chromosome length for size optimisation
chrom_len_ponds_size = int(ponds_stat['area_steps'].sum())

#=============================================================================#
# STEP 5: OPTIMISATION SETTINGS
#=============================================================================#
# Population size
pop_size = config["pop_size"]

# Maximum generations to terminate optimisation
max_generations = config["max_generations"]

# Stopping rule (checked in STEP 8g): after every generation the area under
# the cost-EAD curve of the population (auc_func, units million £ x million £)
# is compared with the previous generation. If it has decreased by less than
# epsilon, or not decreased at all, for consecutive_limit generations in a
# row, the loop stops before max_generations.
# USER INPUT: epsilon and consecutive_limit control this rule.
# Termination criterion based on marginal decrease in area under the curve
auc = np.zeros(max_generations + 1)  # store area under the curve
auc_diff = np.zeros(max_generations + 1)  # delta AUC for consecutive generations

epsilon = 0.001  # threshold for auc_diff
stable_count = 0  # count of consecutive generations that achieve epsilon
consecutive_limit = 5  # number of consecutive generations to consider optimisation stable

#=============================================================================#
# STEP 6: INITIAL POPULATION
#=============================================================================#
# initial_pop() returns the three linked forms of pop_size random designs
# (see NAMING CONVENTIONS). Two designs are fixed: the second-to-last has no
# ponds (the no-intervention baseline) and the last has every pond at full
# size.
# Based on the location chromosomes population, create unique size chromosomes
i_pop_ponds_locat, i_pop_ponds_size, i_pop_ponds_size_val = cf.initial_pop(
    pop_size, chrom_len_ponds_locat, chrom_len_ponds_size, ponds_gene_start,
                                                            ponds_gene_end)

start_timestamp = pd.Timestamp.now()
# CityCAT simulation and building exposure calculation
# for the initial or parent population
print('\n' 'Simulating initial population')

# Simulate every initial design. For each design citycat_exposure() digs its
# ponds into the DEM, runs CityCAT once per rainfall in rainfall_num, then
# classifies the buildings and works out their flood damage. It returns:
#   i_chrom_ponds_volume  volume of each pond (m3), shape (pop_size, ponds)
#   i_chrom_cost          pond life cycle cost (million £), (pop_size, 1)
#   i_expo_high/_medium   buildings per exposure class, (pop_size, rainfalls)
#   _                     low-exposure counts (not used)
#   i_chrom_damage_rps    direct damage in whole £, (pop_size, rainfalls)
[i_chrom_ponds_volume,
 i_chrom_cost, i_expo_high, i_expo_medium, _,
 i_chrom_damage_rps] = cf.citycat_exposure(bldg_file,
                     chrom_len_ponds_locat,
                     pop_size, run_path,
                     rainfall_num, config_num, i_pop_ponds_size_val,
                     ponds_grid_shp, ponds_depth, ponds_grid_cell_area,
                     ponds_area_step_val,
                     pond_unit_vol_lcc, catch_shp,
                     dem_metadata, dem_data,
                     shp_field, cell_index,
                     buffer_list, 0, res_dd, nonres_dd)

# Expected Annual Damage (EAD) is the area under the damage-probability
# curve. Columns 0-4 of i_chrom_damage_rps are the direct damages for the
# return periods T = 10, 20, 30, 50 and 100 years, i.e. annual exceedance
# probabilities P = 1/T. i_d_inf extends the curve to P = 0 by continuing
# the straight line through the 50-year and 100-year points.
i_d_inf = (i_chrom_damage_rps[:, 4] + (i_chrom_damage_rps[:, 4]
                - i_chrom_damage_rps[:, 3]) * (((1/100)-0)/((1/50)-(1/100))))

# EAD by the trapezoidal rule: for each pair of neighbouring return periods,
# (average damage) x (difference in probability), plus the tail from
# P = 1/100 to 0. The result is truncated to whole pounds and converted to
# million £, giving a column of shape (pop_size, 1).
i_chrom_damage = (
    (((i_chrom_damage_rps[:, 0] + i_chrom_damage_rps[:, 1])/2*((1/10)-(1/20))) +
    ((i_chrom_damage_rps[:, 1] + i_chrom_damage_rps[:, 2])/2*((1/20)-(1/30))) +
    ((i_chrom_damage_rps[:, 2] + i_chrom_damage_rps[:, 3])/2*((1/30)-(1/50))) +
    ((i_chrom_damage_rps[:, 3] + i_chrom_damage_rps[:, 4])/2*((1/50)-(1/100))) +
    ((i_chrom_damage_rps[:, 4] + i_d_inf)/2*((1/100)-0))
                                        ).astype(int)/1000000)[:, np.newaxis]

# Where:
# i_chrom_ponds_volume = volume of each pond in a given intervention
# i_chrom_cost = chromosome intervention cost in million £
# i_expo_high = number of highly exposed buildings when the intervention is applied
# i_expo_medium = number of moderately exposed buildings when the intervention is applied
# i_chrom_damage = damage cost in million £ when the intervention is applied

# (i_chrom_damage holds the EAD calculated above, in million £: it is the
# second objective.)
#
# starting value for the stopping rule (STEP 5)
# calculate area under the curve for intial population
auc[0] = cf.auc_func(i_chrom_cost, i_chrom_damage)

# Settings for the cost-EAD scatter plots, saved as PNG files in run_path
# (one per generation).
# fig_background = 'dark_background'
fig_background = 'default'
plot_title = "Initial population (random)"
plot_legend_series = "Evolving solution"
plot_legend_series_1 = "Good solution"
plot_legend_series_2 = "Evolving solution"

# For T = 60 multi return optimisation
plot_x_limit = config["plot_x_limit"]  # [-0.05, 3.1] in the published study
plot_y_limit = config["plot_y_limit"]  # [5.84, 6.4] for 60y30m
plot_x_axis_label = "Intervention cost (million £)"
# NOTE (known issue): this label says "Direct damage cost", but the values
# plotted are EAD (i_chrom_damage, f_chrom_damage and opt_chrom_damage).
plot_y_axis_label = "Direct damage cost (million £)"
# saved as Generation_No_0.png (matplotlib adds .png)
save_file = "Generation_No_0"

cf.scatter_plot(fig_background, plot_title,
                 i_chrom_cost, i_chrom_damage,
                 plot_legend_series,
                    plot_x_limit, plot_y_limit, plot_x_axis_label,
                    plot_y_axis_label, save_file)

#=============================================================================#
# STEP 7: RECORD KEEPING AND OUTPUT LABELS
#=============================================================================#
# These variables keep a record of all unique simulated chromosomes
# g_counter / s_g_counter hold the generation number of every row of
# generation_data.csv / simulated_data.csv (0 = initial population).
onetime_counter = np.zeros(pop_size).astype(int)[:, np.newaxis]
onetime_counter[:, 0] = 0
g_counter = copy.deepcopy(onetime_counter)
s_g_counter = copy.deepcopy(onetime_counter)
simulated_population_size = copy.deepcopy(i_pop_ponds_size)
simulated_population_size_val = copy.deepcopy(i_pop_ponds_size_val)
simulated_chrom_ponds_volume = copy.deepcopy(i_chrom_ponds_volume)
simulated_chrom_cost = copy.deepcopy(i_chrom_cost)
simulated_expo_high = copy.deepcopy(i_expo_high)
simulated_expo_medium = copy.deepcopy(i_expo_medium)
simulated_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
simulated_chrom_damage = copy.deepcopy(i_chrom_damage)

# Labels to export data
# Column labels of the three output CSVs (n = number of ponds):
#   Generation, Size_Steps_Pond_0 ... n-1 (size values),
#   Pond_Volume_0 ... n-1 (m3), Cost (million £),
#   DD_... (direct damage per rainfall, £), EAD (million £).
# That is 2n + 8 columns in total.
pond_count = 0
ponds_columns_length = simulated_chrom_ponds_volume.shape[1]
exp_labels = [None]*(chrom_len_ponds_locat + ponds_columns_length + 8)
for i in range(len((exp_labels))):
    if i > 0 and i < chrom_len_ponds_locat + 1:
        exp_labels[i] = "Size_Steps_Pond_" + str(i-1)
    elif i == 0:
        exp_labels[i] = "Generation"
    elif (i > chrom_len_ponds_locat and
          i < chrom_len_ponds_locat + ponds_columns_length + 1):
        exp_labels[i] = "Pond_Volume_" + str(pond_count)
        pond_count += 1
    elif i == chrom_len_ponds_locat + ponds_columns_length + 1:
        exp_labels[i] = "Cost"
    # NOTE (known issue): the DD_ labels below say 1030 ... 10030, but the
    # default rainfall_num is 1060 ... 10060. The values follow the order of
    # rainfall_num; only the column names differ.
    elif i == chrom_len_ponds_locat + ponds_columns_length + 2:
        exp_labels[i] = "DD_1030"
    elif i == chrom_len_ponds_locat + ponds_columns_length + 3:
        exp_labels[i] = "DD_2030"
    elif i == chrom_len_ponds_locat + ponds_columns_length + 4:
        exp_labels[i] = "DD_3030"
    elif i == chrom_len_ponds_locat + ponds_columns_length + 5:
        exp_labels[i] = "DD_5030"
    elif i == chrom_len_ponds_locat + ponds_columns_length + 6:
        exp_labels[i] = "DD_10030"
    elif i == chrom_len_ponds_locat + ponds_columns_length + 7:
        exp_labels[i] = "EAD"

# These variables keep each generation’s population and objective records
# Store in lists
# (generation 0 = the initial population)
gen_population_size = copy.deepcopy(i_pop_ponds_size)
gen_population_size_val = copy.deepcopy(i_pop_ponds_size_val)
gen_chrom_ponds_volume = copy.deepcopy(i_chrom_ponds_volume)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_expo_high = copy.deepcopy(i_expo_high)
gen_expo_medium = copy.deepcopy(i_expo_medium)
gen_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
gen_chrom_damage = copy.deepcopy(i_chrom_damage)

# Per-generation statistics, keyed by generation number. They are kept in
# memory only (not written to a file): finish time, number of offspring
# simulated, number of offspring that survived into the next population and
# number of designs in the first front of parents + offspring.
gen_time = {}
gen_offspring_count = {}
gen_offspring_sustained_count = {}
gen_front1_count = {}

# Parent population, cost, exposure and damage
# will change in each iteration
p_population_size = copy.deepcopy(i_pop_ponds_size)
p_population_locat = copy.deepcopy(i_pop_ponds_locat)
p_population_size_val = copy.deepcopy(i_pop_ponds_size_val)
p_chrom_ponds_volume = copy.deepcopy(i_chrom_ponds_volume)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_expo_high = copy.deepcopy(i_expo_high)
p_expo_medium = copy.deepcopy(i_expo_medium)
p_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
p_chrom_damage = copy.deepcopy(i_chrom_damage)

#=============================================================================#
# STEP 8: GENERATION LOOP
#=============================================================================#
#-------------------------------------#
""" Generation loop starts here """
#-------------------------------------#

for generation in range(1, max_generations + 1):

    print('\n' f'Gen.{generation}: Generating offspring population')
    #-------------------------------------------------------------------------#
    # 8a  Rank the parents
    #-------------------------------------------------------------------------#
    # p_chroms_obj_record: shape (pop_size, 2); column 0 = cost, column 1 = EAD
    # (both million £, both minimised).
    # Pair of objectives
    # Obj1 size = pop_size x 1, Obj2 size = pop_size x 1
    p_chroms_obj_record = np.concatenate((p_chrom_cost,
                                          p_chrom_damage), axis=1)

    # Rank individuals using dominance depth
    # p_front: {front number: [row numbers]}; front 0 is the non-dominated
    # (Pareto) set
    p_front = cf.non_dominated_sorting(pop_size, p_chroms_obj_record)

    # Maintain diversity using normalised Manhattan distance
    p_distance = cf.calculate_crowding_distance(p_front, p_chroms_obj_record)

    # Sort population by fitness based on front rank and crowding distance
    # row numbers ordered from fittest to least fit: by front first, then by
    # crowding distance (largest first) within each front
    sorted_fitness = np.array(cf.fitness_sort(p_distance, pop_size))

    #-------------------------------------------------------------------------#
    # 8b  Create offspring
    #-------------------------------------------------------------------------#
    # Offspring are created two at a time until there are pop_size designs that
    # have never been simulated (or 5000 attempts have been made).
    #   offspring_size      size chromosomes, (n_offspring, chrom_len_ponds_size)
    #   offspring_size_val  size values, (n_offspring, chrom_len_ponds_locat)
    # Generate offspring
    offspring_size = np.empty((0, chrom_len_ponds_size)).astype(int)
    offspring_size_val = np.empty((0, chrom_len_ponds_locat)).astype(int)

    print('Offspring creation loop counter')
    c = 0  # counter to ensure the while loop is definite
    while len(offspring_size_val) < pop_size and c < 5000:
        # Binary tournament: each parent is the fitter of two randomly drawn
        # designs (see fitter_parent()).
        # Create parents
        parent1_idx = cf.fitter_parent(sorted_fitness, pop_size)
        parent2_idx = cf.fitter_parent(sorted_fitness, pop_size)
        # Ensure parents are different
        while parent1_idx == parent2_idx:
            parent2_idx = cf.fitter_parent(sorted_fitness, pop_size)

        # size chromosomes and location chromosomes of the two parents
        parent1 = p_population_size[parent1_idx, :]
        parent2 = p_population_size[parent2_idx, :]
        
        parent1_locat = p_population_locat[parent1_idx, :]
        parent2_locat = p_population_locat[parent2_idx, :]

        # USER INPUT: genetic operator settings.
        #   co_probability: chance that each gene of a child is taken from the
        #                   other parent (uniform crossover)
        #   m_probability:  chance that each gene is flipped (0 <-> 1), for
        #                   location genes and size genes respectively
        # Crossover probability
        co_probability = 0.6
        # Introduce diversity using the mutation operator
        m_probability = np.array([0.02, 0.02])  # mutation probabilities
        
        # ch1, ch2: the children's size chromosomes; ch1_size_val, ch2_size_val:
        # their size values. Ponds switched off in a child's location chromosome
        # have all their size genes set to 0.
        [ch1, ch2, ch1_size_val, ch2_size_val] = (
            cf.crossover_uniform_mutation_bitflip(parent1, parent2,
                                             parent1_locat, parent2_locat,
                                             co_probability, m_probability,
                                             chrom_len_ponds_locat,
                                             chrom_len_ponds_size,
                                             ponds_gene_start,
                                             ponds_gene_end))

        # reshape to single rows so they can be stacked onto the offspring arrays
        offspring1 = ch1.reshape(1, -1)
        offspring2 = ch2.reshape(1, -1)
        
        offspring1_size_val = ch1_size_val.reshape(1, -1)
        offspring2_size_val = ch2_size_val.reshape(1, -1)
        
        # If either new child has the same pond sizes as an offspring collected
        # earlier, remove that earlier copy; then append both children.
        if len(offspring_size_val) > 0:
            a = []
            for i in range(len(offspring_size_val)):
                # Check if either child represents the same pond settings
                if ((np.all(offspring_size_val[i, :] ==
                                        offspring1_size_val) == True) or
                    (np.all(offspring_size_val[i, :] ==
                                        offspring2_size_val) == True)):
                    a.append(i)
            
            offspring_size_val = np.delete(offspring_size_val, a, 0)
            offspring_size = np.delete(offspring_size, a, 0)
            
            offspring_size_val = np.concatenate((offspring_size_val,
                   offspring1_size_val, offspring2_size_val), axis=0)            
            offspring_size = np.concatenate((offspring_size,
                                   offspring1, offspring2), axis=0)            
        else:
             offspring_size_val = np.concatenate((offspring_size_val,
                    offspring1_size_val, offspring2_size_val), axis=0) 
             offspring_size = np.concatenate((offspring_size,
                                    offspring1, offspring2), axis=0)
             
        # Drop offspring whose pond sizes were already simulated in an earlier
        # generation (including the initial population), so CityCAT never runs
        # twice for the same design.
        offspring_size_val, offspring_size = cf.remove_duplicate_ponds_settigs(
            offspring_size_val, offspring_size, simulated_population_size_val,
            simulated_population_size)

        print(c + 1)
        c = c + 1
    del c

    # Keep at most pop_size offspring; stop the whole run if no new design
    # could be found.
    if len(offspring_size_val) > pop_size:
        offspring_size_val = offspring_size_val[0:pop_size]
        offspring_size = offspring_size[0:pop_size]
        print('\n' f'Gen.{generation}: {len(offspring_size_val)} new offspring found')
    elif len(offspring_size_val) > 0 and len(offspring_size_val) < pop_size:
        print('\n' f'Gen.{generation}: Only {len(offspring_size_val)} new offspring found')
    elif len(offspring_size_val) == 0:
        print('\n' f'Gen.{generation}: Could not find new offspring')
        sys.exit(0)
    else:
        print('\n' f'Gen.{generation}: {len(offspring_size_val)} new offspring found')

    gen_offspring_count[generation] = len(offspring_size_val)
    #-------------------------------------------------------------------------#
    # 8c  Simulate the offspring
    #-------------------------------------------------------------------------#
    # CityCAT simulation and building exposure calculation
    # for the offspring population
    print('\n' f'Gen.{generation}: Simulating offspring population')
    
    # Same call as in STEP 6, for the offspring only.
    [o_chrom_ponds_volume,
     o_chrom_cost,
     o_expo_high,
     o_expo_medium, _,
     o_chrom_damage_rps] = cf.citycat_exposure(bldg_file,
                         chrom_len_ponds_locat,
                         len(offspring_size_val), run_path,
                         rainfall_num, config_num, offspring_size_val,
                         ponds_grid_shp, ponds_depth, ponds_grid_cell_area,
                         ponds_area_step_val,
                         pond_unit_vol_lcc, catch_shp,
                         dem_metadata, dem_data,
                         shp_field, cell_index,
                         buffer_list, generation, res_dd, nonres_dd)

    # EAD of each offspring, calculated exactly as in STEP 6 (see there).
    o_d_inf = (o_chrom_damage_rps[:, 4] + (o_chrom_damage_rps[:, 4]
                    - o_chrom_damage_rps[:, 3]) * (((1/100)-0)/((1/50)-(1/100))))

    o_chrom_damage = (
    (((o_chrom_damage_rps[:, 0] + o_chrom_damage_rps[:, 1])/2*((1/10)-(1/20))) +
     ((o_chrom_damage_rps[:, 1] + o_chrom_damage_rps[:, 2])/2*((1/20)-(1/30))) +
     ((o_chrom_damage_rps[:, 2] + o_chrom_damage_rps[:, 3])/2*((1/30)-(1/50))) +
     ((o_chrom_damage_rps[:, 3] + o_chrom_damage_rps[:, 4])/2*((1/50)-(1/100))) +
     ((o_chrom_damage_rps[:, 4] + o_d_inf)/2*((1/100)-0))
                                         ).astype(int)/1000000)[:, np.newaxis]
    
    #-------------------------------------------------------------------------#
    # 8d  Save every simulated design
    #-------------------------------------------------------------------------#
    # Append the offspring to the record of every design simulated so far.
    # Save unique chromosomes created in each generation
    simulated_population_size_val = np.concatenate(
                (simulated_population_size_val, offspring_size_val), axis=0)
    simulated_population_size = np.concatenate((simulated_population_size,
                                           offspring_size), axis=0)    
    simulated_chrom_ponds_volume = np.concatenate(
                (simulated_chrom_ponds_volume, o_chrom_ponds_volume), axis=0)
    simulated_chrom_cost = np.concatenate((simulated_chrom_cost,
                                           o_chrom_cost), axis=0)
    simulated_expo_high = np.concatenate((simulated_expo_high,
                                           o_expo_high), axis=0)
    simulated_expo_medium = np.concatenate((simulated_expo_medium,
                                            o_expo_medium), axis=0)
    simulated_chrom_damage_rps = np.concatenate((simulated_chrom_damage_rps, 
                                           o_chrom_damage_rps), axis=0)
    simulated_chrom_damage = np.concatenate((simulated_chrom_damage,
                                           o_chrom_damage), axis=0)

    # Important note: the simulated population and its objectives represent
    # the offspring created in each generation. Do not mix with the
    # generation wise best population.

    # simulated_data.csv is rewritten after every generation, so the results
    # so far are kept if a run stops early. Columns: see exp_labels (STEP 7).
    # Export simulated data
    onetime_counter = np.zeros(len(offspring_size_val)
                                               ).astype(int)[:, np.newaxis]
    onetime_counter[:, 0] = generation
    s_g_counter = np.concatenate((s_g_counter, onetime_counter), axis=0)
    simulated_output = np.empty((0, chrom_len_ponds_locat + 
                                                 ponds_columns_length + 8))
    simulated_output = np.concatenate((s_g_counter,
                                       simulated_population_size_val,
                                       simulated_chrom_ponds_volume,
                                       simulated_chrom_cost,
                                       simulated_chrom_damage_rps,
                                       simulated_chrom_damage),
                                       axis=1)

    simulated_df = pd.DataFrame(simulated_output, columns=exp_labels)
    simulated_df.to_csv('simulated_data.csv', index_label='SN')

    #-------------------------------------------------------------------------#
    # 8e  Combine parents and offspring and keep the best
    #-------------------------------------------------------------------------#
    # Pair of offspring objectives
    o_chroms_obj_record = np.concatenate((o_chrom_cost,
                                          o_chrom_damage), axis=1)

    # Combine parents’ and offspring objectives
    # comb_ arrays: the parents' rows first, followed by the offspring's rows.
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record, o_chroms_obj_record), axis=0)

    # Code checkpoint
    # Check duplicate records in the combined objective list
    # Designs whose (cost, EAD) pair exactly repeats an earlier row are removed.
    # dup_idx_obj lists their row numbers; the same rows are deleted from every
    # other comb_ array below, so all arrays stay aligned row by row.
    [comb_chroms_obj_record_uniq, dup_idx_obj] = cf.remove_duplicate_list(
                                            comb_chroms_obj_record)

    # Join parent and offspring individuals
    comb_pop_size_val = np.concatenate((p_population_size_val,
                                               offspring_size_val), axis=0)
    comb_pop_size = np.concatenate((p_population_size, offspring_size), axis=0)

    # Remove chromosomes corresponding to duplicate objectives
    comb_pop_size_val_uniq_obj = cf.remove_same_objectives_population(
                                            comb_pop_size_val, dup_idx_obj)
    
    comb_pop_size_uniq_obj = cf.remove_same_objectives_population(
                                            comb_pop_size, dup_idx_obj)

    # Note: the name comb_pop_size is reused here. From now on it is the number
    # of unique combined designs (an int), not the array of size chromosomes.
    comb_pop_size = len(comb_pop_size_val_uniq_obj)
    # Elitist selection (NSGA-II): parents and offspring are ranked together and
    # the best pop_size designs are kept, so good parents are never lost.
    # Rank individuals from the combined population
    comb_front = cf.non_dominated_sorting(comb_pop_size,
                                   comb_chroms_obj_record_uniq)

    # Calculate crowding distance for the combined population
    comb_distance = cf.calculate_crowding_distance(comb_front,
                                            comb_chroms_obj_record_uniq)

    # Sort combined population based on ranking and crowding distance
    comb_population_fitness_sort = cf.fitness_sort(comb_distance, comb_pop_size)

    # Select the fittest individuals. As individuals are already sorted,
    # select the first pop_size items
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])

    # Remove the duplicate-objective rows from every other combined array too,
    # so the row numbers in select_fittest point to the same design everywhere.
    comb_chrom_ponds_volume = np.concatenate((p_chrom_ponds_volume,
                                              o_chrom_ponds_volume), axis=0)
    comb_chrom_ponds_volume_uniq_obj = np.delete(comb_chrom_ponds_volume,
                                                     dup_idx_obj, 0)

    # Joined cost objective of parent and offspring populations
    comb_chrom_cost = np.concatenate((p_chrom_cost, o_chrom_cost), axis=0)
    comb_chrom_cost_uniq_obj = np.delete(comb_chrom_cost, dup_idx_obj, 0)
    
    # Joined exposure objectives of parent and offspring populations
    comb_expo_high = np.concatenate((p_expo_high, o_expo_high), axis=0)
    comb_expo_high_uniq_obj = np.delete(comb_expo_high, dup_idx_obj, 0)
    comb_expo_medium = np.concatenate((p_expo_medium, o_expo_medium), axis=0)
    comb_expo_medium_uniq_obj = np.delete(comb_expo_medium, dup_idx_obj, 0)

    comb_chrom_damage_rps = np.concatenate(
                        (p_chrom_damage_rps, o_chrom_damage_rps), axis=0)
    comb_chrom_damage_rps_uniq_obj = np.delete(
                                        comb_chrom_damage_rps, dup_idx_obj, 0)

    comb_chrom_damage = np.concatenate((p_chrom_damage, o_chrom_damage), axis=0)
    comb_chrom_damage_uniq_obj = np.delete(comb_chrom_damage, dup_idx_obj, 0)

    # f_ arrays: the fittest pop_size designs, in fitness order (best first).
    # Select objectives for the fittest individuals
    f_chrom_ponds_volume = copy.deepcopy(
                            comb_chrom_ponds_volume_uniq_obj[select_fittest])
    f_chrom_cost = copy.deepcopy(comb_chrom_cost_uniq_obj[select_fittest])
    f_expo_high = copy.deepcopy(comb_expo_high_uniq_obj[select_fittest])
    f_expo_medium = copy.deepcopy(comb_expo_medium_uniq_obj[select_fittest])
    f_chrom_damage_rps = copy.deepcopy(comb_chrom_damage_rps_uniq_obj[select_fittest])
    f_chrom_damage = copy.deepcopy(comb_chrom_damage_uniq_obj[select_fittest])

    # Select the fittest individuals to create the new population
    f_population_size_val = copy.deepcopy(comb_pop_size_val_uniq_obj
                                                      [select_fittest])
    f_population_size = copy.deepcopy(comb_pop_size_uniq_obj[select_fittest])

    # scatter plot of the new population: Generation_No_<generation>.png
    plot_title = "Generation No " + str(generation)
    save_file = "Generation_No_" + str(generation)

    cf.scatter_plot(fig_background, plot_title, f_chrom_cost,
                        f_chrom_damage, plot_legend_series,
                        plot_x_limit, plot_y_limit, plot_x_axis_label,
                        plot_y_axis_label, save_file)

    #-------------------------------------------------------------------------#
    # 8f  Make the fittest designs the new parents and record them
    #-------------------------------------------------------------------------#
    # Make a copy of the previous population
    old_population_size_val = copy.deepcopy(p_population_size_val)

    # Separate new chromosomes from old ones by comparing the new population
    # with the old population
    # NOTE (known issue): separate_new_old() raises UnboundLocalError if none of
    # the parents survives into the new population (f_old_chroms_index is only
    # set when at least one is kept). Its results are only used for the counts
    # printed below and stored in gen_offspring_sustained_count.
    [new_chroms, old_chroms, old_chroms_index] = cf.separate_new_old(
                            f_population_size_val, p_population_size_val)
    print('\n' f'Gen.{generation}: New population contains {len(old_chroms)} parents and {len(new_chroms)} offspring')
    
    # Delete old population variables
    del (p_population_size, p_population_size_val, p_population_locat,
         p_chrom_cost, p_expo_high,
         p_expo_medium, p_chrom_damage_rps, p_chrom_damage, p_chrom_ponds_volume)

    # The fittest designs become the parents of the next generation.
    # Assign new population
    p_population_size_val = copy.deepcopy(f_population_size_val)
    p_population_size = copy.deepcopy(f_population_size)
    # rebuild the location chromosomes from the size values: a pond is on (1)
    # if at least one of its size steps is switched on
    p_population_locat = np.where(p_population_size_val != 0, 1, 0)
    p_chrom_ponds_volume = copy.deepcopy(f_chrom_ponds_volume)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_expo_high = copy.deepcopy(f_expo_high)
    p_expo_medium = copy.deepcopy(f_expo_medium)
    p_chrom_damage_rps = copy.deepcopy(f_chrom_damage_rps)
    p_chrom_damage = copy.deepcopy(f_chrom_damage)

    # Append the new population to the per-generation records (gen_ arrays).
    gen_population_size_val = np.concatenate((gen_population_size_val,
                                           p_population_size_val), axis=0)    
    gen_population_size = np.concatenate((gen_population_size,
                                           p_population_size), axis=0)
    gen_chrom_ponds_volume = np.concatenate((gen_chrom_ponds_volume,
                                           p_chrom_ponds_volume), axis=0)
    gen_chrom_cost = np.concatenate((gen_chrom_cost,
                                           p_chrom_cost), axis=0)
    gen_expo_high = np.concatenate((gen_expo_high,
                                           p_expo_high), axis=0)
    gen_expo_medium = np.concatenate((gen_expo_medium,
                                           p_expo_medium), axis=0)
    gen_chrom_damage_rps = np.concatenate((gen_chrom_damage_rps, 
                                           p_chrom_damage_rps), axis=0)
    gen_chrom_damage = np.concatenate((gen_chrom_damage,
                                           p_chrom_damage), axis=0)

    # per-generation statistics (see STEP 7)
    gen_offspring_sustained_count[generation] = len(new_chroms)
    gen_front1_count[generation] = len(comb_front[0])
    gen_time[generation] = pd.Timestamp.now()

    # generation_data.csv: every generation's population (pop_size rows each;
    # generation 0 = initial population), rewritten after every generation.
    # Export generation data
    onetime_counter = np.zeros(pop_size).astype(int)[:, np.newaxis]
    onetime_counter[:, 0] = generation
    g_counter = np.concatenate((g_counter, onetime_counter), axis=0)
    generation_output = np.empty((0, chrom_len_ponds_locat +
                                              ponds_columns_length + 8))
    generation_output = np.concatenate((g_counter,
                                       gen_population_size_val,
                                       gen_chrom_ponds_volume,
                                       gen_chrom_cost,
                                       gen_chrom_damage_rps,
                                       gen_chrom_damage),
                                       axis=1)

    generation_df = pd.DataFrame(generation_output, columns=exp_labels)
    generation_df.to_csv('generation_data.csv', index_label='SN')

    #-------------------------------------------------------------------------#
    # 8g  Stopping rule
    #-------------------------------------------------------------------------#
    # auc_diff > 0 means the trade-off curve moved down (improved). A generation
    # counts as stable when the improvement is below epsilon; consecutive_limit
    # stable generations in a row end the loop early (see STEP 5).
    # Termination criterion check
    auc[generation] = cf.auc_func(p_chrom_cost, p_chrom_damage)
    auc_diff[generation] = auc[generation-1] - auc[generation]

    if auc_diff[generation] < epsilon:
        stable_count += 1
        if stable_count >= consecutive_limit:
            print(f"Termination criterion met at generation {generation}")
            break   # now correctly inside the loop
    else:
        stable_count = 0

#=============================================================================#
# STEP 9: FINAL RESULT (PARETO-OPTIMAL DESIGNS)
#=============================================================================#
# Get optimal chromosomes or solutions

# Make pairs of objectives for the final population
opt_chroms_objs = np.concatenate((p_chrom_cost, p_chrom_damage), axis=1)

# The first front of the final population is the set of optimal (Pareto)
# designs: none of them is beaten on both cost and EAD by another design.
# Get nondominated solutions, that is the first front
opt_front = cf.non_dominated_sorting(pop_size,
                               opt_chroms_objs)[0]

# Population that provides the optimal solutions
opt_population_size_val = p_population_size_val[opt_front]
opt_chrom_ponds_volume = p_chrom_ponds_volume[opt_front]

# Optimal cost
opt_chrom_cost = p_chrom_cost[opt_front]

# Direct damages by return period for the optimal solutions
opt_chrom_damage_rps = p_chrom_damage_rps[opt_front]

# Optimal risk
opt_chrom_damage = p_chrom_damage[opt_front]

# Buildings for optimal risk
opt_expo_high = p_expo_high[opt_front]
opt_expo_medium = p_expo_medium[opt_front]

# generation is the last generation run (lower than max_generations if the
# stopping rule ended the loop early)
# Plot title
opt_plot_title = 'Generation no ' + str(generation) + ' optimal'

# Plot legend
opt_plot_legend_series = 'Optimal solution'

# Output file name
opt_save_file = 'Generation_No_' + str(generation) + '_optimal'

cf.scatter_plot(fig_background, opt_plot_title, opt_chrom_cost, opt_chrom_damage,
                 opt_plot_legend_series,
                    plot_x_limit, plot_y_limit, plot_x_axis_label,
                    plot_y_axis_label, opt_save_file)

# Export optimal data
# (this empty array is replaced by the concatenation below)
opt_output = np.empty((0, chrom_len_ponds_size + ponds_columns_length + 8))

# Final generation number
opt_g_counter = np.zeros(len(opt_front)).astype(int)[:, np.newaxis]
opt_g_counter[:, 0] = generation

# same columns as the other CSVs (labels in exp_labels, STEP 7)
# Create export array for optimal data
opt_output = np.concatenate((opt_g_counter, opt_population_size_val,
                                   opt_chrom_ponds_volume,
                                   opt_chrom_cost,
                                   opt_chrom_damage_rps,
                                   opt_chrom_damage,),
                                   axis=1)

# Create a dataframe from the array
opt_df = pd.DataFrame(opt_output, columns=exp_labels)

# optimised_data.csv in run_path: one row per optimal design
# Export optimal data as a CSV file
opt_df.to_csv('optimised_data.csv', index_label='SN')

#-----------------------------------------------------------------------------#
""" THE END """

#-----------------------------------------------------------------------------#
