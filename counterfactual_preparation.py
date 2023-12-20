'''
Paul Voit 21 Nov 2022
THis version is written to work with GPKG-files to process Germany as a whole
This script contains all the functions which are used in the workflow of deriving the hydrographs for subbasins
'''

import pandas as pd
import pcraster as pcr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import datetime
import os
import rioxarray as rxr
import geopandas as gpd
from itertools import combinations
import shutil

#######################
#Prepare basic rasters
#######################

def create_basic_files(input_path):
    '''
    This is the multiprocessing version. Only difference is that it takes a tuple as input and unpacks it.
    This function makes sure that all necessary files for the traveltime computation are there and otherwise creates
    them. The only two files needed at the beginning are (unfilled) DEM of the region and the raster in the same dimension
    which contains the mannings-n-values.

    :param input_path: str, path to input file folder
    :param fill: the fill amount for filling the dem with PCRaster
    :return: creates and writes all the necessary files for the calculation hydrographs
    '''

    fill = 1e31

    if not os.path.exists(f'output/gis/dem.map'):
        #Todo: change to Python function
        os.system(
            f'gdal_translate -of "PCRaster" -a_srs EPSG:3035 {input_path}dem.tif output/gis/dem.map'
        )

    if not os.path.exists(f'{input_path}dem_filled.map'):
        print('Filling DEM')
        pcr.setclone('output/gis/dem.map')
        dem = pcr.readmap('output/gis/dem.map')
        dem_filled = pcr.lddcreatedem(dem, fill, fill, fill, fill)
        pcr.report(dem_filled, f'output/gis/dem_filled.map')

    if not os.path.exists(f'output/gis/flowdir.map'):
        print('Creating flow direction grid')
        pcr.setclone(f'output/gis/dem_filled.map')
        dem = pcr.readmap(f'output/gis/dem_filled.map')
        flowdir = pcr.lddcreate(dem, fill, fill, fill, fill)
        pcr.report(flowdir, f'output/gis/flowdir.map')

    if not os.path.exists(f'output/gis/accu.map'):
        print('Creating accumulation grid')
        pcr.setclone('output/gis/dem.map')
        flowdir = pcr.readmap(f'output/gis/flowdir.map')
        accu = pcr.accuflux(flowdir, 1)
        pcr.report(accu, f'output/gis/accu.map')

    if not os.path.exists(f'output/gis/subbasins.gpkg'):
        print("Creating subbasin shape file and raster")
        accu = pcr.readmap(f'output/gis/accu.map')
        flowdir = pcr.readmap(f'output/gis/flowdir.map')

        s_order = get_stream_order(input_path, save=True)
        outlets, outlet_array = create_outlet_raster(s_order, accu)

        subbasins = pcr.subcatchment(flowdir, pcr.nominal(outlets))
        subbasins_raster, outlets, size_df = remove_small_subbasins(
            subbasins, flowdir, outlet_array, thresh=700)

        pcr.report(outlets, f'output/gis/outlets.map')
        pcr.report(subbasins_raster, f'output/gis/subbasins_adjusted.map')

        #change to Python function
        os.system(
            f'gdal_polygonize.py output/gis/subbasins_adjusted.map -f "GPKG" output/gis/subbasins_first.gpkg'
        )
        subs = gpd.read_file(f'output/gis/subbasins_first.gpkg')
        subs['geometry'] = subs.buffer(0)  # this fixes the geometries
        subs = subs.dissolve(by="DN")  # join single part polygons with same ID
        subs = subs.loc[1:, :]
        subs = subs.set_crs("epsg:3035", allow_override=True)
        subs.to_file(f"output/gis/subbasins.gpkg")

        os.remove(f'output/gis/subbasins_first.gpkg')

    if not os.path.exists(f'output/gis/friction_maidment.map'):
        dem_filled = pcr.readmap(f'output/gis/dem_filled.map')
        accu = pcr.readmap(f'output/gis/accu.map')
        subbasin_raster = pcr.readmap(f'output/gis/subbasins_adjusted.map')
        print("Making maidment friction map")
        slopearea = make_slopearea_raster(dem_filled, accu, input_path)
        make_friction_map(subbasin_raster,
                          slopearea,
                          input_path,
                          lower_limit=0.06,
                          upper_limit=3)

    if not os.path.exists(f'output/gis/tt_complete.map'):

        pcr.setclone(f'output/gis/dem_filled.map')
        flowdir = pcr.readmap(f'output/gis/flowdir.map')
        outlets = pcr.readmap(f'output/gis/outlets.map')
        friction = pcr.readmap(
            f'output/gis/friction_maidment.map')

        outlets_vals = pcr.pcr2numpy(outlets, -99)

        print(f'Number of outlets: {len(np.unique(outlets_vals))}')
        position = outlets_vals > 0
        position[position] = 1
        position = position.astype(int)
        outlet_raster = pcr.numpy2pcr(pcr.Boolean, position, -99)

        print(f"{datetime.datetime.now()} Creating traveltime raster")
        tt_complete = pcr.ldddist(flowdir, outlet_raster, friction)
        tt_complete = tt_complete / 60
        pcr.report(tt_complete, f'output/gis/tt_complete.map')

    if not os.path.exists(
            f'output/gis/soil_landuse_classes.gpkg'):
        print(f"{datetime.datetime.now()} Creating soil landuse shape file for curve number method")
        make_cn_soil_landuse_shapefile(input_path)


def make_slopearea_raster(dem, accu, input_path):
    '''
    Creates the slopearea raster which is needed to make the friction raster
    :param dem: pcraster._pcraster.Field, filled DEM
    :param accu: pcraster._pcraster.Field, accumulation raster
    :param input_path: path to the input files
    :return: pcraster._pcraster.Field. slopearea raster
    '''

    a = 0.5
    b = 0.5

    slopefraction = pcr.slope(dem)
    slopearea = ((slopefraction * 100)**a) * (accu**b)

    pcr.report(slopearea, f'output/gis/slopearea.map')

    return slopearea


def make_friction_map(basin_raster, slopearea, input_path,
                      lower_limit, upper_limit):
    '''
    Calculate the friction map for all the subbasins accrodin to the method of Maidment et al. (1996)
    :param basin_raster:  pcraster._pcraster.Field: raster containing the subbasin IDs
    :param slopearea: pcraster._pcraster.Field: the slopearea raster computed by the function make_slopearea_raster. Slopearea is the slope of a
    cell divided by its accumulation value.

    :param input_path: string, path to the folder containing the rasters
    :param lower_limit: lower limit of the velocity value.
    :param upper_limit: upper limit of the velocity value
    :return: writes a pcraster._pcraster.Field containing the friction values which are needed for the computation of
    hydrograph.
    '''

    basins_nominal = pcr.nominal(basin_raster)
    #get the average slopearea for each subbasin
    mean_slopearea = pcr.areaaverage(slopearea, basins_nominal)
    pcr.report(mean_slopearea, f"output/gis/mean_slopearea.map")

    v_unlimited = 0.1 * (slopearea / mean_slopearea)

    # limit the velocity
    v_limited = pcr.ifthenelse((v_unlimited > upper_limit),
                               pcr.scalar(upper_limit), v_unlimited)
    v_limited = pcr.ifthenelse((v_unlimited < lower_limit),
                               pcr.scalar(lower_limit), v_limited)

    friction = 1 / v_limited

    pcr.report(friction, f"output/gis/friction_maidment.map")


def make_cn_soil_landuse_shapefile(input_path):
    '''
    Creates a shapefile with Curvenumbers for the whole main basin from a landuse shapefile containing CORINE codes
    and a soil shapefile which contains soils classes.
    This shapefile will be used to calculate the direct runoff with the SCS-CN method.
    :param ger_main_basins_shape: shapefile which contains the main basins of Germany
    :param ger_soils_shape: german wide soil classes (A,B,C,D) shapefile. This was derived manually from the BUEK250
    :param landuse_shape: Corine landuse codes for the main basins (shapefile)
    :param cn_codes: file contains the keys for the translation of CORINE codes into CN landuse codes (four for each
    landuse class, for each class four soil types)
    :return: writes  a shapefile which then can be used for calculation of direct runoff via the SCS-CN method
    '''

    soils_shape = gpd.read_file(
        f'{input_path}buek250_cn_classes.gpkg')

    cn_codes = pd.read_csv(f'{input_path}scs.csv',
                           sep='\t')

    landuse = gpd.read_file(f'{input_path}corine.gpkg')
    landuse.CLC18 = landuse.CLC18.astype(int)
    landuse = pd.merge(landuse, cn_codes, on="CLC18")
    landuse.rename(columns={" SCS_B": "SCS_B"}, inplace=True)

    #intersect the two
    intersect = gpd.overlay(soils_shape, landuse, how="intersection")
    intersect = intersect.drop(
        ["SCHRAFFUR", "LAND", "NRKART", "SYM_NR", "leg"], axis=1)

    intersect["area km2"] = intersect.geometry.area / 1e6
    intersect.to_file(f'output/gis/soil_landuse_classes.gpkg')



def get_stream_order(input_path, save=True):
    '''
    Creates and saves the streams and their order (Strahler) to raster
    :param input_path: str, path to input file folder
    :param save: save the stream grid
    :return: pcraster._pcraster.Field containing the streams and their order
    '''

    pcr.setclone(f'output/gis/flowdir.map')
    flowdir = pcr.readmap(f'output/gis/flowdir.map')
    s_order = pcr.streamorder(flowdir)

    max_s = pcr.mapmaximum(s_order)
    max_s = pcr.cellvalue(max_s, 0, 0)
    max_s = max_s[0]

    if save:
        for order in range(5, max_s + 1):
            stream = pcr.ifthen(s_order == order, s_order)
            pcr.report(stream, f"output/gis/stream_{order}.map")

        streams_all = pcr.ifthen(s_order > 3, s_order)
        pcr.report(streams_all, f"output/gis/streams_all.map")

    return s_order


def find_intersects(streamorder, accu, order1, order2):
    '''
    This function finds all the cells where a cell of order1 and a cell of order2 intersect. These points are then
    used as outlets for the subbasin creation.
    This function is called by create_outlet_raster().

    :param streamorder: pcraster._pcraster.Field, grid containing the stream order for each cell. Provided by the function
    get_stream_order.
    :param accu: pcraster._pcraster.Field, grid containing the accumulation values for each cell
    :param order1: int, order 1
    :param order2: int, order 2
    :return: np.array containing the points where there are intersections between rivers of Strahler order 7-10
    '''

    assert order2 > order1
    s_vals = pcr.pcr2numpy(streamorder, 0)
    accu_vals = pcr.pcr2numpy(accu, np.nan)
    s_vals[~np.logical_or(s_vals == order1, s_vals == order2)] = 0
    kernel = np.ones((3, 3))
    padded_image = np.pad(s_vals, (1, 1))
    window_sum = convolve2d(padded_image, kernel, mode='valid')

    cond1 = (window_sum % order2 > 0) & (window_sum > 2 * order1) & (
        window_sum % order1 > 0)  # True where neighbors exist
    cond2 = s_vals == order1  # True for cells with streamorder 6

    test = cond1 & cond2
    intersect_points = np.argwhere(test)

    # add additional points to split larger subs because sometimes a very small river flows into a very big one.
    # This code is a bit ugly, there could be a nicer way to iterate through all the cases....
    for i in [9, 10, 11, 12, 13]:

        if (order1 >= 7 and order2 == i):
            for point in intersect_points:

                neighbors = s_vals[point[0] - 1:point[0] + 2,
                                   point[1] - 1:point[1] + 2]
                neighbors_accu = accu_vals[point[0] - 1:point[0] + 2,
                                           point[1] - 1:point[1] + 2]

                if np.any(neighbors == i):

                    neighbors = neighbors == i
                    neighbors_accu[np.logical_not(neighbors)] = np.nan

                    # take the value which is stream order 10 but has the lowest accumulation
                    coord = np.where(
                        neighbors_accu == np.nanmin(neighbors_accu))

                    if coord[0][0] == 0:
                        x = -1
                    if coord[0][0] == 1:
                        x = 0
                    if coord[0][0] == 2:
                        x = 1

                    if coord[1][0] == 0:
                        y = -1
                    if coord[1][0] == 1:
                        y = 0
                    if coord[1][0] == 2:
                        y = 1

                    point[0] = point[0] + x
                    point[1] = point[1] + y

                    intersect_points = np.concatenate(
                        (intersect_points, point.reshape((1, 2))), axis=0)

    return intersect_points


def create_outlet_raster(s_order, accu):
    '''
    For the creation of subbasins we need to define outlets. We do this by finding the intersection points
    of rivers with different orders. The output of this function can be used for pcr.subcatchment()
    :param s_order: pcraster._pcraster.Field, grid containing the stream order for each cell
    :param accu: pcraster._pcraster.Field, grid containing the accumulation values for each cell
    :return: pcraster._pcraster.Field, grid containing the labeled outlets
            np.array, array containing the labeled outlets
    '''
    # define at which points subbasins should be created
    max_s = pcr.mapmaximum(s_order)
    max_s = pcr.cellvalue_by_indices(max_s, 0, 0)[0]

    orders = [i for i in range(7, (max_s + 1))]
    combs = combinations(orders, 2)

    combination_list = [i for i in combs]

    intersects = find_intersects(s_order, accu, combination_list[0][0],
                                 combination_list[0][1])

    for combination in combination_list[1:]:
        dummy = find_intersects(s_order, accu, combination[0], combination[1])
        intersects = np.concatenate((intersects, dummy), axis=0)

    labels = [i for i in range(1, intersects.shape[0] + 1)]
    vals = pcr.pcr2numpy(s_order, 0)
    dummy = np.zeros((vals.shape[0], vals.shape[1]))

    for i in range(intersects.shape[0]):
        dummy[intersects[i][0], intersects[i][1]] = labels[i]

    # add max outlet
    test = pcr.pcr2numpy(accu, np.nan)
    coords = np.argwhere(test == np.nanmax(test))

    dummy[coords[0][0], coords[0][1]] = intersects.shape[0] + 1

    outlets = pcr.numpy2pcr(pcr.Scalar, dummy, mv=0)

    return outlets, dummy


def remove_small_subbasins(subbasins, flowdir, outlet_array, thresh=700):
    '''

    :param subbasins: pcraster._pcraster.Field, grid which contains the subbasins with labels
    :param flowdir: pcraster._pcraster.Field, grid containing the flowdirection in each cell (derived with pcr.ldddist())
    :param outlet_array: numpy.array containing the labeled outlets, output of create_outlet_raster
    :param thresh: int, outlets whose subbasins contain less cells than the threshold will be removed
    :return: subbasins_new, pcraster._pcraster.Field, grid containing the new subbasins
            outlets_new, pcraster._pcraster.Field, grid containing new outlets
            size_df, a dataframe containing the sizes in km² of the new subbasins
    '''

    subs = pcr.pcr2numpy(subbasins, 0)
    outlet_array_new = np.zeros((outlet_array.shape[0], outlet_array.shape[1]))

    sub_ids = subs[subs > 0]
    sub_ids = sub_ids.flatten()
    sub_ids = list(set(sub_ids))

    size_df = pd.DataFrame({'id': sub_ids, 'size_km2': 0.0})

    # find small subbasins and remove the outlet points of these
    count = 0
    label = 1

    for id in sub_ids:
        size = np.count_nonzero(subs == id)
        size_df.loc[size_df.id == id, "size_km2"] = float((size * 25 * 25) / 1e6)

        if size < thresh:
            count = count + 1

        else:
            outlet_array_new[outlet_array == id] = label
            label = label + 1

    print(f"Removed {count} outlets")

    # calculate subbasins again with removed outlet points
    outlets_new = pcr.numpy2pcr(pcr.Nominal, outlet_array_new, mv=0)
    subbasins_new = pcr.subcatchment(flowdir, outlets_new)

    print(f"The basin now contains {label - 1} subbasins")

    return subbasins_new, outlets_new, size_df


def flowdir_to_coord(flowdir_val):
    '''
    Translates the flowdirection of a cell to a vector. If this vector is added to the
    coordinate of the cell, we get the coordinate of the cell into which the initial flows.
    PCRaster labels flowdirection clockwise, like on the num pad. 8=up, 2=down, 4=left, 6=right etc...
    :param flowdir_val: int, flowdirection
    :return: tuple, containing the vector to add to the cell coordinate
    '''

    # tuple(x,y)
    if flowdir_val == 8:
        index = (0, 25)
    if flowdir_val == 9:
        index = (25, 25)
    if flowdir_val == 6:
        index = (25, 0)
    if flowdir_val == 3:
        index = (25, -25)
    if flowdir_val == 2:
        index = (0, -25)
    if flowdir_val == 1:
        index = (-25, -25)
    if flowdir_val == 4:
        index = (-25, 0)
    if flowdir_val == 7:
        index = (-25, 25)
    if flowdir_val == 5:
        index = (0, 0)
        print("Attention, outlet is a pit!")

    return index


#######################
#Clip to subbasins
#######################

def make_cn_calculator():
    '''
    Builds a calculator, which is able to calculate the correct curve number for soil moisture class 1 and three 3
    any curve number for soil moisture class 2.
    :return: dictionary, CN-calculator
    '''
    sm_classes = pd.read_csv(f'input/cn_sm_classes.csv')
    fit1 = np.polyfit(sm_classes["Soil moisture class 2"].to_numpy(),
                      sm_classes["Soil moisture class 1"].to_numpy(), 2)
    fit3 = np.polyfit(sm_classes["Soil moisture class 2"].to_numpy(),
                      sm_classes["Soil moisture class 3"].to_numpy(), 2)

    yfitted1 = np.poly1d(fit1)
    yfitted3 = np.poly1d(fit3)

    cn_calculator = {"1": yfitted1, "3": yfitted3}

    return cn_calculator


def make_outlet_df(tt_raster, subbasin_id):
    '''

    :param tt_raster: xarray.core.dataarray.DataArray, raster containing the traveltime (in minutes) to the outlet.
    :param subbasin_id: int, subbasin to be processed
    :return: writes the coordinates (inidices) of the outlet to file. This is needed later for hydrological modelling.
    '''

    try:
        outlet_coords = np.where(tt_raster == 0)
        outlet_df = pd.DataFrame({'y': [outlet_coords[1][0]], 'x': [outlet_coords[2][0]]})
        outlet_df.to_csv(f'output/basins/{subbasin_id}/outlet_{subbasin_id}.csv', index=False)
        return outlet_df
    except:

        print(f'Subbasin {subbasin_id} is an incomplete basin on the side and has no outlet.')
        return None


def get_giuh(tt_raster, subbasin_id, delta_t=15, plot=True):
    '''
    Calculates the hydrograph (GIUH) from the traveltime raster.
    :param tt_raster: xarray.core.dataarray, raster containing the traveltime (in minutes) to the outlet
    :param subbasin_id: int, id of subbasin to be processed
    :param delta_t: int, temporal resolution of the hydrograph
    :param plot: boolean, if true, generate plots
    :return: Writes the hydrograph to output/basins/hydrograph_xx.csv
             Writes the plot of the hydrograph to output/giuh_plots/xxx_giuh.png
    '''

    tt_vals = tt_raster.values
    tt_vals_flat = tt_vals.flatten()
    tt_vals_flat = tt_vals_flat[~np.isnan(tt_vals_flat)]

    bins = np.arange(0, max(tt_vals_flat), delta_t)
    inds = np.digitize(tt_vals_flat, bins)

    count_list = np.bincount(inds)

    # transform to discharge (m3/s), cellsize 25mx25m, rain=1mm (l/m2/h) ????
    count_list = (count_list * 25 * 25 * 1) / (delta_t * 60) / 1000  # 15 = dt

    # write unit hydrograph to file
    # #+ 15, so the first value is 15. This means that all values which arrive between 0 and 15 belong in this bin.
    # similar to rainfall. The rainfall at 15:00 describes the rainfall from 14:00-15:00
    # to achieve this the bin list has to be extended, because of the shift to the left
    bins = bins[1:]
    extend = np.array([bins[-1] + delta_t, bins[-1] + 2 * delta_t])
    bins = np.concatenate([bins, extend])
    hydrograph = pd.DataFrame({
        'delta_t': bins[:len(count_list)],
        'discharge_m3/s': count_list
    })
    hydrograph["discharge_m3/s"] = hydrograph["discharge_m3/s"].round(2)
    hydrograph.to_csv(f'output/basins/{subbasin_id}/hydrograph_{subbasin_id}.csv',
                      index=False)


    if plot:
        if not os.path.exists('output/giuh_plots'):
            os.mkdir('output/giuh_plots')

        outlet_df = pd.read_csv(f'output/basins/{subbasin_id}/outlet_{subbasin_id}.csv')
        x_coord = outlet_df.loc[0, "x"]
        y_coord = outlet_df.loc[0, "y"]

        fig = plt.figure(figsize=(22, 10))
        axs = fig.add_subplot(121)
        # add data to plots
        axs.plot(hydrograph['delta_t'], hydrograph['discharge_m3/s'])
        axs.set_xlabel("time (min)")
        axs.set_ylabel("discharge $m^3/s$")
        axs.set_title("GIUH")
        axs = fig.add_subplot(122)
        basinplot = axs.imshow(tt_vals[0, :, :])
        plt.title(f"Sub {subbasin_id}")
        axs.scatter(x_coord, y_coord, s=150, marker="x", color="red")
        fig.colorbar(basinplot, shrink=0.6)
        axs.set_title("Traveltime to outlet (min)")
        plt.savefig(f"output/giuh_plots/{subbasin_id}_giuh.png", bbox_inches="tight")
        plt.close()

def get_cn_df(cn_sub, subbasin_id, cn_calculator):
    '''
    Calculates one curve number for the whole subbasin with the areal fraction in cn_sub. Then the according curve numbers
    for soil moisture class 1 and 3 are calculated. The results are returned in a dataframe
    :param cn_sub: geopandas.Dataframe, containing the soil-landuse classes for the subbasins
    :param subbasin_id: int, subbasin to be processed
    :param cn_calculator: dictionary, the curve number calculator generated with make_cn_calculator
    :return: pandas.Dataframe, contains the curve numbers for all soil moisture classes for the subbasins.
    '''

    #merge areas with same curve number
    cn_sub["CN"] = -99
    for soil_class in ["A", "B", "C", "D"]:
        cn_sub.loc[cn_sub.cn_soil_cl == soil_class,  ["CN", "CN_for_dissolve"]] = cn_sub.loc[cn_sub.cn_soil_cl == soil_class,  f"SCS_{soil_class}"].copy()

    cn_sub = cn_sub.dissolve(by="CN_for_dissolve")
    cn_sub = cn_sub[["CN", "geometry"]]
    cn_sub["area_km2"] = cn_sub.area / 1e6

    cn_sub.to_file(f'output/basins/{subbasin_id}/cn_{subbasin_id}.gpkg')

    # make dataframe which contains the three CN for the whole basin (based on the areal fractions of the different
    # landuse-soil classes), to read later when calculating the effective rainfall
    cn_class_df = pd.DataFrame({"sub_id": [subbasin_id]})
    cn_class_df["cn_sm_1"] = int(round((cn_calculator["1"](cn_sub.CN) * (
                cn_sub["area_km2"] / cn_sub["area_km2"].sum())).sum(), 0))
    cn_class_df["cn_sm_2"] = int(
        round((cn_sub.CN * (cn_sub["area_km2"] / cn_sub["area_km2"].sum())).sum(), 0))
    cn_class_df["cn_sm_3"] = int(round((cn_calculator["3"](cn_sub.CN) * (
                cn_sub["area_km2"] / cn_sub["area_km2"].sum())).sum(), 0))

    return cn_class_df


def hydrograph_and_cn_for_subbasins(subbasin_list, plot_giuh=True):
    '''
    Clip the traveltime, and the curve number shapefile to the individual subbasins contained in subbasin_list.
    From the clipped traveltime raster the geomorphological instantenous unit hydrograph (GIUH) is computed and stored.
    The coordinates of the subbasin outlet, as well as the curve number for the subbasin is stored together with the
    hydrograph in output/basins/xxx.
    The hydrographs are plotted and stored in output/giuh_plot.
    A table of curve numbers for the al basins contained in basin_list is stored in
    output/gis/CN_subbasins_table.csv
    These files are needed later for hydrological modelling.
    This function calls get_giuh and get_cn_df
    :param subbasin_list: list, containing all subbasins to be processed
    :param plot_giuh: boolean, whether to plot the hydrographs or not. Slows down the computation
    :return: writes traveltime raster, hydrograph, outlet coordinates and curve for every individual subbasin in
    output/basins/xxx.
    Plots of the hydrographs are stored in output/giuh_plots.
    Writes the curve number table for all subbasins in output/gis/CN_subbasins.table.csv
    '''

    tt_complete = rxr.open_rasterio('output/gis/tt_complete.map',
                                    mask_and_scale=True)  # masked nodata to np.nan
    tt_complete = tt_complete.rio.write_crs("EPSG:3035")

    subs = gpd.read_file("output/gis/subbasins.gpkg")

    cn_class_df = pd.DataFrame({"sub_id": [-99], "cn_sm_1": [-99], "cn_sm_2": [-99], "cn_sm_3": [-99]})

    cn_calculator = make_cn_calculator()

    if not os.path.exists("output"):
        os.mkdir("output")

    if not os.path.exists("output/basins"):
        os.mkdir("output/basins")

    for subbasin_id in subbasin_list:

        print(f'Creating hydrograph for subbasin {subbasin_id}')
        if os.path.exists(f'output/basins/{subbasin_id}'):
            print(f'Files for subbasins {subbasin_id} already exist. Not generating new files.')
            print('If you want to newly calculate the subbasin files, the old one have to be deleted first.')
            continue


        if not os.path.exists(f'output/basins/{subbasin_id}'):
            os.mkdir(f'output/basins/{subbasin_id}')

        mask = subs.loc[subs.DN == subbasin_id, :].copy()

        #clip the traveltime raster for the subbasin
        clipped = tt_complete.rio.clip(mask['geometry'])
        clipped.rio.to_raster(f'output/basins/{subbasin_id}/tt_{subbasin_id}.tif')


        # clip the soil-landuse classes for the subbasin
        cn_sub = gpd.read_file('output/gis/soil_landuse_classes.gpkg',mask=mask['geometry'])
        cn_sub = cn_sub.clip(mask["geometry"])

        print(f"Writing outlet dataframe for subbasin {subbasin_id}")
        outlet = make_outlet_df(clipped, subbasin_id)

        if outlet is None:
            continue

        print(f'Calculating hydrograph for subbasin {subbasin_id}')
        get_giuh(tt_raster=clipped, subbasin_id=subbasin_id, delta_t=15, plot=plot_giuh)

        print(f'Calculating curve number for subbasin {subbasin_id}')
        cn_df = get_cn_df(cn_sub, subbasin_id, cn_calculator)

        cn_class_df = pd.concat([cn_class_df, cn_df])

    cn_class_df = cn_class_df.loc[cn_class_df.sub_id != -99, :]

    print("Writing CN table for all basins.")
    cn_class_df.to_csv(f'output/gis/CN_subbasins_table.csv', index=False)


def get_floworder():
    '''
    For each subbasin, it is checked into which subbasin it flows using the outlet, and the flow direction at this point.
    Subbasins where the calculation of the traveltime was not succesful (these are usually subbasins which are on the
    sides and situated in the buffer) are removed and a new shapefile ("subbasins_info.shp) is written.
    Then the network is analysed: Subbasins which don't have an inflow are head basins and have the order "1". All the
    basins which have as inflow just head catchments get the order "2" we iterate through this process until we found
    the order of every subbasin. This order then will be used to superposition the unit hydrographs in the right order.

    :return: writes a cleaned up shapefile and a dataframe ("flow_order.csv") which contains the floworder for all
            subbasins.
            writes a shapefile which contains all the flow order info for all the subbasins ("subbasins_info.gpkg")

    '''

    outlets = rxr.open_rasterio(f'output/gis/outlets.map')
    subbasins = rxr.open_rasterio(f'output/gis/subbasins_adjusted.map')
    subbasins_shape = gpd.read_file(f'output/gis/subbasins.gpkg')
    flowdir = rxr.open_rasterio(f'output/gis/flowdir.map')
    accu = rxr.open_rasterio(f'output/gis/accu.map')
    flowdir_vals = flowdir.values[0, :, :]
    outlet_vals = outlets.values[0, :, :]
    sub_ids = subbasins_shape.DN.to_list()
    df = pd.DataFrame({'sub_id': sub_ids, 'flows_to': 0, 'accu_at_outlet': 0, 'time_to_next_outlet': 0})
    problematic_subs = []


    for id in sub_ids:
        print(f"Sub {id}")

        outlet_loc = np.where(outlet_vals == id)
        x_outlet = float(outlets.x[outlet_loc[1][0]])
        y_outlet = float(outlets.y[outlet_loc[0][0]])

        flowdir_at_outlet = flowdir_vals[outlet_vals == id]

        flows_to_coordinate = flowdir_to_coord(flowdir_at_outlet)

        x_inlet = x_outlet + flows_to_coordinate[0]
        y_inlet = y_outlet + flows_to_coordinate[1]

        flows_to = subbasins.sel(x=x_inlet, y=y_inlet).values[0]

        if flows_to != 0:
            next_basin = rxr.open_rasterio(f'output/basins/{flows_to}/tt_{flows_to}.tif')
            next_basin_size = np.count_nonzero(subbasins[0, :, :].values == flows_to)

            if next_basin.shape[1] * next_basin.shape[2] < 0.6 * next_basin_size:
                print(f"Traveltime raster for Sub {flows_to} problematic")
                tt_to_next_outlet = -999
                problematic_subs.append(flows_to)
            else:
                tt_to_next_outlet = int(round(next_basin.sel(x=x_inlet, y=y_inlet).values[0], 0))
        else:
            next_basin = 0
            tt_to_next_outlet = -999

        df.loc[df.sub_id == id, 'flows_to'] = flows_to
        df.loc[df.sub_id == id, 'accu_at_outlet'] = accu.sel(x=x_outlet, y=y_outlet).values[0]
        df.loc[df.sub_id == id, 'time_to_next_outlet'] = tt_to_next_outlet

    problematic_subs = list(set(problematic_subs))
    print(f'Nr. Problematic subs: {len(problematic_subs)}')

    # clean up shapefile and remove the problematic basins, which are mostly because of the buffer
    sub_shape = gpd.read_file('output/gis/subbasins.gpkg')

    # remove also the subbasins where the traveltime didn't work. these are usually within the buffer
    print("Checking all the subbasins for their size. If 40 % NaN, we remove it")
    for id in sub_ids:
        basin = rxr.open_rasterio(f'output/basins/{id}/tt_{id}.tif')
        basin_size = np.count_nonzero(subbasins[0, :, :].values == id)

        if basin.shape[1] * basin.shape[2] < 0.6 * basin_size:
            print(f"Traveltime raster for Sub {id} problematic")
            problematic_subs.append(id)

    sub_shape = sub_shape[~sub_shape['DN'].isin(problematic_subs)]

    print("Writing cleaned up subbasin file")
    print(f'Nr. Problematic subs: {len(problematic_subs)}')

    sub_shape.to_file('output/gis/subbasins_cleaned.gpkg')

    # this df also needs to be filtered with the problematic subs
    df = df[~df['sub_id'].isin(problematic_subs)]

    ## get the flow order
    ## find the head basins
    print("Getting floworder")
    df.loc[:, "order"] = np.nan

    sub_list = df.sub_id.values.tolist()

    all_subs = set(sub_list)
    with_inflow = set(df.flows_to.values.tolist())

    head_catchments = all_subs - with_inflow
    head_catchments = list(head_catchments)

    # level 1 = head_catchments
    df.loc[df.sub_id.isin(head_catchments), 'order'] = 1

    df.inflow_basins = "0"

    for i in range(len(df)):
        sub_id = df.iloc[i, 0]
        dummy = df.loc[df.flows_to == sub_id, "sub_id"].values.tolist()

        if len(dummy) != 0:
            df.loc[df.sub_id == sub_id, 'inflow_basins'] = str(dummy)

    # get calculation order in loop:
    # level 1 = head_catchments
    df.loc[df.sub_id.isin(head_catchments), 'order'] = 1
    upstream_complete = head_catchments

    # find the ones which flow to themselves (pits)
    for i in df.index:
        if df.loc[i, 'sub_id'] == df.loc[i, 'flows_to']:
            df.loc[i, 'order'] = -99
            print(f'Subbasin {df.loc[i, "sub_id"]} contains a pit')

    order = 2
    order_list = [1]

    while any(np.isnan(df.loc[:, 'order'].values)):

        print(f'Order: {order}')

        for i in range(len(df)):
            sub_id = df.iloc[i, 0]
            test = df.loc[df.sub_id == sub_id, 'inflow_basins'].values[0]

            if not isinstance(test, str):
                continue

            if np.isnan(df.loc[df.sub_id == sub_id, 'order'].values[0]):

                test = test.replace('[', '')
                test = test.replace(']', '')
                test = test.split(',')
                test = [int(i) for i in test]

                if all([i in upstream_complete for i in test]):
                    df.loc[df.sub_id == sub_id, 'order'] = order

            else:
                continue

        upstream_complete = df.loc[~np.isnan(df.order), 'sub_id'].values.tolist()
        order_list.append(order)
        order = order + 1
        print(f'{np.count_nonzero(np.isnan(df.order.values))} subs remaining')

    df.to_csv('output/gis/flow_order.csv', index=False)

    mb = gpd.read_file(f'output/gis/subbasins_cleaned.gpkg')

    sub_ids = mb.DN.to_list()

    df = df.loc[df.sub_id.isin(sub_ids), :]

    gdf = pd.merge(mb, df, left_on="DN", right_on="sub_id")
    gdf.drop(columns=["DN"], inplace=True)
    gdf["area_km2"] = round(gdf.area / 1e6, 2)
    gdf["cum_upstream_area"] = gdf.area_km2

    orders = list(set(gdf.order.to_list()))

    res_list = list()
    res_list.append(gdf.loc[gdf.order == 1, :])

    #get the cumulated upstream area for each basin/outlet

    for i in orders[1:]:
        dummy = gdf.loc[gdf.order == i, :]
        #dummy.cum_upstream_area = dummy.apply(calculate_upstream_area, axis=1)

        for id in dummy.sub_id:

            upstream_basins = dummy.loc[dummy.sub_id == id, "inflow_basins"].values[0]
            upstream_basins = upstream_basins.strip('[')
            upstream_basins = upstream_basins.strip(']')
            upstream_basins = upstream_basins.split(', ')

            area = dummy.loc[dummy.sub_id == id, "area_km2"].values[0]

            for basin in upstream_basins:
                try:
                    area = area + gdf.loc[gdf.sub_id == int(basin), "cum_upstream_area"].values[0]
                except:
                    continue

            gdf.loc[gdf.sub_id == id, "cum_upstream_area"] = area

    gdf.to_file(f'output/gis/subbasins_info.gpkg')
    os.remove(f'output/gis/subbasins_cleaned.gpkg')

    just_table = gdf.drop("geometry", axis=1)
    just_table.to_csv(f'output/gis/subbasins_info_table.csv')

#######################
#Extract rainfall data
#######################



