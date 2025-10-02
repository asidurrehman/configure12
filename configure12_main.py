# -*- coding: utf-8 -*-
"""
╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝

File name: configure12_main.py

@author: Asid Ur Rehman
Exposure analysis adapted from Robert Berstch
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
import importlib.util

warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------#
# Creating and setting folders and paths
#-----------------------------------------------------------------------------#
# CityCAT run path
run_path = os.path.join('C:', os.path.sep, 'z', 'configure12')

configure_func_path = os.path.join(run_path, 'codes',
                                   'configure12_functions.py')
# Load configure_func
spec1 = importlib.util.spec_from_file_location("configure_func", configure_func_path)
cf = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(cf)

# Change working directory
os.chdir(run_path)
# Show working directory
print("Current working directory: {0}".format(os.getcwd()))
# CityCAT rainfall file numbers
rainfall_num = np.array([1060, 2060, 3060, 5060, 10060])
# CityCAT configuration file number
config_num = 3

# Buildings shapefile
bldg_file = os.path.join(run_path, 'vector', 'buildings.shp')

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
df_res_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'residential_damage_calculation.csv')))
# Dataframe to numpy array
res_dd = np.array(df_res_dd_data)

# For nonresidential
df_nonres_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'non_residential_damage_calculation.csv')))
# Dataframe to numpy array
nonres_dd = np.array(df_nonres_dd_data)

#-----------------------------------------------------------------------------#
""" Main code starts here """
#-----------------------------------------------------------------------------#
# Get variables needed to calculate building exposure in each
# GA generation
[shp_field, cell_index, buffer_list] = cf.spatial_indices(bldg_file, run_path,
                                       idx_rainfall, idx_config, csv_file)

# Read ponds shapefile
ponds_grid_shp = gpd.read_file(os.path.join(run_path,
                                    'vector', 'all_possible_ponds_grid.shp'))
# Life cycle cost of a pond (25 years)
# Life cycle cost is calculated as pounds per cubic metre
pond_unit_vol_lcc = 153.2
# One time construction cost
pond_unit_vol_copex = 89.42
# Operations and maintenance cost uses inflated values

# To ensure the DEM created after modification is not displaced,
# convert the DEM extent to a shapefile
catch_shp = gpd.read_file(os.path.join(run_path, 'vector', 'catchment_dem.shp'))

dem_original_path = os.path.join(run_path, 'domain_dem_original',
                                 'Domain_DEM.asc')

# Read DEM ASCII file
dem_metadata, dem_data = cf.read_dem_ascii(dem_original_path)

# Count number of ponds
ponds_count = ponds_grid_shp['Id'].nunique()

# Compute the minimum elevation for each object (pond) ID
min_elv = ponds_grid_shp.groupby('Id')['elevation'].min().reset_index()

# Rename the column in the geodataframe
min_elv.rename(columns={'elevation': 'min_elv'}, inplace=True)

# Merge the minimum elevation back into the original dataframe
ponds_grid_shp = ponds_grid_shp.merge(min_elv, on='Id')

# Calculate elevation removal needed to level the pond base
ponds_grid_shp["elv_diff"] = (ponds_grid_shp["min_elv"] -
                              ponds_grid_shp["elevation"])

ponds_grid_shp["pond_depth"] = -1
# Negative value because this will be subtracted from the DEM

ponds_grid_shp["pond_cell_total_depth"] = (ponds_grid_shp["elv_diff"] +
                          ponds_grid_shp["pond_depth"])

# Calculate area
ponds_grid_shp['area'] = ponds_grid_shp.geometry.area

# Volume to remove for pond base levelling
ponds_grid_shp["lev_vol"] = (abs(ponds_grid_shp["elv_diff"]) *
                                                     ponds_grid_shp['area'])

# Calculate area of each pond in square metres
ponds_stat = ponds_grid_shp.groupby('Id')['area'].sum().reset_index()
ponds_area = np.array(ponds_stat['area'])

# Depth of ponds in metres. The user can assign different depths to different ponds
ponds_stat['pond_depth'] = 1
ponds_depth = np.array(ponds_stat['pond_depth'])[np.newaxis, :]

# Maximum volume in cubic metres
ponds_stat['max_vol'] = ponds_stat['area'] * ponds_stat['pond_depth']

# Minimum pond area and related volume to consider
ponds_stat['min_area'] = 100
ponds_stat['min_volume'] = ponds_stat['min_area'] * ponds_stat['pond_depth']

# Area steps for each pond
ponds_stat['area_steps'] = (ponds_stat['area'] /
                             ponds_stat['min_area']).astype(int)

# Define starting positions of genes for each pond
ponds_stat['gene_end'] = ponds_stat['area_steps'].cumsum()
ponds_stat['gene_start'] = ponds_stat['gene_end'] - ponds_stat['area_steps']

ponds_gene_start = np.array(ponds_stat['gene_start'])
ponds_gene_end = np.array(ponds_stat['gene_end'])

# Depth step value for each pond
ponds_grid_cell_area = 25
ponds_stat['grid_cell_area'] = ponds_grid_cell_area
ponds_stat['area_step_val'] = ponds_stat['min_area'] / ponds_stat['grid_cell_area']
# 25 is the pond grid cell area
ponds_area_step_val = np.array(ponds_stat['area_step_val'])[np.newaxis, :]

# Chromosome length for location optimisation
chrom_len_ponds_locat = ponds_count

# Chromosome length for size optimisation
chrom_len_ponds_size = int(ponds_stat['area_steps'].sum())

# Population size
pop_size = 100

# Maximum generations to terminate optimisation
max_generations = 50

# Termination criterion based on marginal decrease in area under the curve
auc = np.zeros(max_generations + 1)  # store area under the curve
auc_diff = np.zeros(max_generations + 1)  # delta AUC for consecutive generations

epsilon = 0.001  # threshold for auc_diff
stable_count = 0  # count of consecutive generations that achieve epsilon
consecutive_limit = 5  # number of consecutive generations to consider optimisation stable

# Based on the location chromosomes population, create unique size chromosomes
i_pop_ponds_locat, i_pop_ponds_size, i_pop_ponds_size_val = cf.initial_pop(
    pop_size, chrom_len_ponds_locat, chrom_len_ponds_size, ponds_gene_start,
                                                            ponds_gene_end)

start_timestamp = pd.Timestamp.now()
# CityCAT simulation and building exposure calculation
# for the initial or parent population
print('\n' 'Simulating initial population')

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

i_d_inf = (i_chrom_damage_rps[:, 4] + (i_chrom_damage_rps[:, 4]
                - i_chrom_damage_rps[:, 3]) * (((1/100)-0)/((1/50)-(1/100))))

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

# calculate area under the curve for intial population
auc[0] = cf.auc_func(i_chrom_cost, i_chrom_damage)

# fig_background = 'dark_background'
fig_background = 'default'
plot_title = "Initial population (random)"
plot_legend_series = "Evolving solution"
plot_legend_series_1 = "Good solution"
plot_legend_series_2 = "Evolving solution"

# For T = 60 multi return optimisation
plot_x_limit = [-0.05, 3.1]
plot_y_limit = [5.84, 6.4]  # for 60y30m
plot_x_axis_label = "Intervention cost (million £)"
plot_y_axis_label = "Direct damage cost (million £)"
save_file = "Generation_No_0"

cf.scatter_plot(fig_background, plot_title,
                 i_chrom_cost, i_chrom_damage,
                 plot_legend_series,
                    plot_x_limit, plot_y_limit, plot_x_axis_label,
                    plot_y_axis_label, save_file)

# These variables keep a record of all unique simulated chromosomes
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
gen_population_size = copy.deepcopy(i_pop_ponds_size)
gen_population_size_val = copy.deepcopy(i_pop_ponds_size_val)
gen_chrom_ponds_volume = copy.deepcopy(i_chrom_ponds_volume)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_expo_high = copy.deepcopy(i_expo_high)
gen_expo_medium = copy.deepcopy(i_expo_medium)
gen_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
gen_chrom_damage = copy.deepcopy(i_chrom_damage)

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

#-------------------------------------#
""" Generation loop starts here """
#-------------------------------------#

for generation in range(1, max_generations + 1):

    print('\n' f'Gen.{generation}: Generating offspring population')
    # Pair of objectives
    # Obj1 size = pop_size x 1, Obj2 size = pop_size x 1
    p_chroms_obj_record = np.concatenate((p_chrom_cost,
                                          p_chrom_damage), axis=1)

    # Rank individuals using dominance depth
    p_front = cf.non_dominated_sorting(pop_size, p_chroms_obj_record)

    # Maintain diversity using normalised Manhattan distance
    p_distance = cf.calculate_crowding_distance(p_front, p_chroms_obj_record)

    # Sort population by fitness based on front rank and crowding distance
    sorted_fitness = np.array(cf.fitness_sort(p_distance, pop_size))

    # Generate offspring
    offspring_size = np.empty((0, chrom_len_ponds_size)).astype(int)
    offspring_size_val = np.empty((0, chrom_len_ponds_locat)).astype(int)

    print('Offspring creation loop counter')
    c = 0  # counter to ensure the while loop is definite
    while len(offspring_size_val) < pop_size and c < 5000:
        # Create parents
        parent1_idx = cf.fitter_parent(sorted_fitness, pop_size)
        parent2_idx = cf.fitter_parent(sorted_fitness, pop_size)
        # Ensure parents are different
        while parent1_idx == parent2_idx:
            parent2_idx = cf.fitter_parent(sorted_fitness, pop_size)

        parent1 = p_population_size[parent1_idx, :]
        parent2 = p_population_size[parent2_idx, :]
        
        parent1_locat = p_population_locat[parent1_idx, :]
        parent2_locat = p_population_locat[parent2_idx, :]

        # Crossover probability
        co_probability = 0.6
        # Introduce diversity using the mutation operator
        m_probability = np.array([0.02, 0.02])  # mutation probabilities
        
        [ch1, ch2, ch1_size_val, ch2_size_val] = (
            cf.crossover_uniform_mutation_bitflip(parent1, parent2,
                                             parent1_locat, parent2_locat,
                                             co_probability, m_probability,
                                             chrom_len_ponds_locat,
                                             chrom_len_ponds_size,
                                             ponds_gene_start,
                                             ponds_gene_end))

        offspring1 = ch1.reshape(1, -1)
        offspring2 = ch2.reshape(1, -1)
        
        offspring1_size_val = ch1_size_val.reshape(1, -1)
        offspring2_size_val = ch2_size_val.reshape(1, -1)
        
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
             
        offspring_size_val, offspring_size = cf.remove_duplicate_ponds_settigs(
            offspring_size_val, offspring_size, simulated_population_size_val,
            simulated_population_size)

        print(c + 1)
        c = c + 1
    del c

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
    # CityCAT simulation and building exposure calculation
    # for the offspring population
    print('\n' f'Gen.{generation}: Simulating offspring population')
    
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

    o_d_inf = (o_chrom_damage_rps[:, 4] + (o_chrom_damage_rps[:, 4]
                    - o_chrom_damage_rps[:, 3]) * (((1/100)-0)/((1/50)-(1/100))))

    o_chrom_damage = (
    (((o_chrom_damage_rps[:, 0] + o_chrom_damage_rps[:, 1])/2*((1/10)-(1/20))) +
     ((o_chrom_damage_rps[:, 1] + o_chrom_damage_rps[:, 2])/2*((1/20)-(1/30))) +
     ((o_chrom_damage_rps[:, 2] + o_chrom_damage_rps[:, 3])/2*((1/30)-(1/50))) +
     ((o_chrom_damage_rps[:, 3] + o_chrom_damage_rps[:, 4])/2*((1/50)-(1/100))) +
     ((o_chrom_damage_rps[:, 4] + o_d_inf)/2*((1/100)-0))
                                         ).astype(int)/1000000)[:, np.newaxis]
    
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

    # Pair of offspring objectives
    o_chroms_obj_record = np.concatenate((o_chrom_cost,
                                          o_chrom_damage), axis=1)

    # Combine parents’ and offspring objectives
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record, o_chroms_obj_record), axis=0)

    # Code checkpoint
    # Check duplicate records in the combined objective list
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

    comb_pop_size = len(comb_pop_size_val_uniq_obj)
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

    plot_title = "Generation No " + str(generation)
    save_file = "Generation_No_" + str(generation)

    cf.scatter_plot(fig_background, plot_title, f_chrom_cost,
                        f_chrom_damage, plot_legend_series,
                        plot_x_limit, plot_y_limit, plot_x_axis_label,
                        plot_y_axis_label, save_file)

    # Make a copy of the previous population
    old_population_size_val = copy.deepcopy(p_population_size_val)

    # Separate new chromosomes from old ones by comparing the new population
    # with the old population
    [new_chroms, old_chroms, old_chroms_index] = cf.separate_new_old(
                            f_population_size_val, p_population_size_val)
    print('\n' f'Gen.{generation}: New population contains {len(old_chroms)} parents and {len(new_chroms)} offspring')
    
    # Delete old population variables
    del (p_population_size, p_population_size_val, p_population_locat,
         p_chrom_cost, p_expo_high,
         p_expo_medium, p_chrom_damage_rps, p_chrom_damage, p_chrom_ponds_volume)

    # Assign new population
    p_population_size_val = copy.deepcopy(f_population_size_val)
    p_population_size = copy.deepcopy(f_population_size)
    p_population_locat = np.where(p_population_size_val != 0, 1, 0)
    p_chrom_ponds_volume = copy.deepcopy(f_chrom_ponds_volume)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_expo_high = copy.deepcopy(f_expo_high)
    p_expo_medium = copy.deepcopy(f_expo_medium)
    p_chrom_damage_rps = copy.deepcopy(f_chrom_damage_rps)
    p_chrom_damage = copy.deepcopy(f_chrom_damage)

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

    gen_offspring_sustained_count[generation] = len(new_chroms)
    gen_front1_count[generation] = len(comb_front[0])
    gen_time[generation] = pd.Timestamp.now()

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

# Get optimal chromosomes or solutions

# Make pairs of objectives for the final population
opt_chroms_objs = np.concatenate((p_chrom_cost, p_chrom_damage), axis=1)

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
opt_output = np.empty((0, chrom_len_ponds_size + ponds_columns_length + 8))

# Final generation number
opt_g_counter = np.zeros(len(opt_front)).astype(int)[:, np.newaxis]
opt_g_counter[:, 0] = generation

# Create export array for optimal data
opt_output = np.concatenate((opt_g_counter, opt_population_size_val,
                                   opt_chrom_ponds_volume,
                                   opt_chrom_cost,
                                   opt_chrom_damage_rps,
                                   opt_chrom_damage,),
                                   axis=1)

# Create a dataframe from the array
opt_df = pd.DataFrame(opt_output, columns=exp_labels)

# Export optimal data as a CSV file
opt_df.to_csv('optimised_data.csv', index_label='SN')

#-----------------------------------------------------------------------------#
""" THE END """
#-----------------------------------------------------------------------------#