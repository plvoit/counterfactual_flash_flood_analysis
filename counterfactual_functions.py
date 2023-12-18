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
import rasterio
import geopandas as gpd
from itertools import combinations


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

    if not os.path.exists(f'{input_path}dem.map'):
        os.system(
            f'gdal_translate -of "PCRaster" -a_srs EPSG:3035 {input_path}dem.tif {input_path}dem.map'
        )

    if not os.path.exists(f'{input_path}dem_filled.map'):
        pcr.setclone(f'{input_path}dem.map')
        dem = pcr.readmap(f'{input_path}dem.map')
        print('Creating filled DEM')
        dem_filled = pcr.lddcreatedem(dem, fill, fill, fill, fill)
        pcr.report(dem_filled, f'{input_path}dem_filled.map')

    if not os.path.exists(f'{input_path}flowdir.map'):
        print('Creating flow direction grid')
        pcr.setclone(f'{input_path}dem_filled.map')
        dem = pcr.readmap(f'{input_path}dem_filled.map')
        flowdir = pcr.lddcreate(dem, fill, fill, fill, fill)
        pcr.report(flowdir, f'{input_path}flowdir.map')

    if not os.path.exists(f'{input_path}accu.map'):
        print('Creating accumulation grid')
        pcr.setclone(f'{input_path}dem.map')
        flowdir = pcr.readmap(f'{input_path}flowdir.map')
        accu = pcr.accuflux(flowdir, 1)
        pcr.report(accu, f'{input_path}accu.map')

    if not os.path.exists(f'{input_path}subbasins.gpkg'):
        print("Creating subbasin shape file and raster")
        accu = pcr.readmap(f'{input_path}accu.map')
        flowdir = pcr.readmap(f'{input_path}flowdir.map')

        s_order = get_stream_order(input_path, save=True)
        outlets, outlet_array = create_outlet_raster(s_order, accu)

        subbasins = pcr.subcatchment(flowdir, pcr.nominal(outlets))
        subbasins_raster, outlets, size_df = remove_small_subbasins(
            subbasins, flowdir, outlet_array, thresh=700)

        pcr.report(outlets, f'{input_path}outlets.map')
        pcr.report(subbasins_raster, f'{input_path}subbasins_adjusted.map')

        os.system(
            f'gdal_polygonize.py {input_path}subbasins_adjusted.map -f "GPKG" {input_path}subbasins_first.gpkg'
        )
        subs = gpd.read_file(f'{input_path}subbasins_first.gpkg')
        subs['geometry'] = subs.buffer(0)  # this fixes the geometries
        subs = subs.dissolve(by="DN")  # join single part polygons with same ID
        subs = subs.loc[1:, :]
        subs = subs.set_crs("epsg:3035", allow_override=True)
        subs.to_file(f"{input_path}subbasins.gpkg")

        os.remove(f'{input_path}subbasins_first.gpkg')

    if not os.path.exists(f'{input_path}friction_maidment.map'):
        dem_filled = pcr.readmap(f'{input_path}dem_filled.map')
        accu = pcr.readmap(f'{input_path}accu.map')
        subbasin_raster = pcr.readmap(f'{input_path}subbasins_adjusted.map')
        print("Making maidment friction map")
        slopearea = make_slopearea_raster(dem_filled, accu, input_path)
        make_friction_map(subbasin_raster,
                          slopearea,
                          input_path,
                          lower_limit=0.06,
                          upper_limit=3)

    if not os.path.exists(f'{input_path}tt_complete.map'):

        pcr.setclone(f'{input_path}dem_filled.map')
        flowdir = pcr.readmap(f'{input_path}flowdir.map')
        outlets = pcr.readmap(f'{input_path}outlets.map')
        friction = pcr.readmap(
            f'{input_path}friction_maidment.map')

        outlets_vals = pcr.pcr2numpy(outlets, -99)

        print(f'Number of outlets: {len(np.unique(outlets_vals))}')
        position = outlets_vals > 0
        position[position] = 1
        position = position.astype(int)
        outlet_raster = pcr.numpy2pcr(pcr.Boolean, position, -99)

        print(f"{datetime.datetime.now()} Starting traveltime")
        tt_complete = pcr.ldddist(flowdir, outlet_raster, friction)
        tt_complete = tt_complete / 60
        pcr.report(tt_complete, f'{input_path}tt_complete.map')
        print(f"{datetime.datetime.now()} Traveltime complete")

    if not os.path.exists(
            f'{input_path}soil_landuse_classes.gpkg'):
        print(f"{datetime.datetime.now()}Creating soil landuse shape file for curve number method")
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

    pcr.report(slopearea, f'{input_path}slopearea.map')

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
    mean_slopearea = pcr.areaaverage(slopearea, basins_nominal)
    pcr.report(mean_slopearea, f"{input_path}mean_slopearea.map")

    v_unlimited = 0.1 * (slopearea / mean_slopearea)

    # limit the velocity
    v_limited = pcr.ifthenelse((v_unlimited > upper_limit),
                               pcr.scalar(upper_limit), v_unlimited)
    v_limited = pcr.ifthenelse((v_unlimited < lower_limit),
                               pcr.scalar(lower_limit), v_limited)

    friction = 1 / v_limited

    pcr.report(friction, f"{input_path}friction_maidment.map")


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
    #ger_soils_shape = ger_soils_shape.to_crs("epsg:3035")

    cn_codes = pd.read_csv(f'{input_path}scs.csv',
                           sep='\t')

    landuse = gpd.read_file(f'{input_path}corine.gpkg')
    landuse.CLC18 = landuse.CLC18.astype(int)
    landuse = pd.merge(landuse, cn_codes, on="CLC18")
    landuse.rename(columns={" SCS_B": "SCS_B"}, inplace=True)

    #intersect the two
    intersect = gpd.overlay(soils_shape, landuse, how="intersection")
    print(intersect.columns)
    intersect = intersect.drop(
        ["SCHRAFFUR", "LAND", "NRKART", "SYM_NR", "leg"], axis=1)

    intersect["area km2"] = intersect.geometry.area / 1e6
    intersect.to_file(f'{input_path}soil_landuse_classes.gpkg')



def get_stream_order(input_path, save=True):
    '''
    Creates and saves the streams and their order (Strahler) to raster
    :param input_path: str, path to input file folder
    :param save: save the stream grid
    :return: pcraster._pcraster.Field containing the streams and their order
    '''

    pcr.setclone(f'{input_path}flowdir.map')
    flowdir = pcr.readmap(f'{input_path}flowdir.map')
    s_order = pcr.streamorder(flowdir)

    max_s = pcr.mapmaximum(s_order)
    max_s = pcr.cellvalue(max_s, 0, 0)
    max_s = max_s[0]

    if save:
        for order in range(5, max_s + 1):
            stream = pcr.ifthen(s_order == order, s_order)
            pcr.report(stream, f"{input_path}stream_{order}.map")

        streams_all = pcr.ifthen(s_order > 3, s_order)
        pcr.report(streams_all, f"{input_path}streams_all.map")

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

    size_df = pd.DataFrame({'id': sub_ids, 'size_km2': 0})

    # find small subbasins and remove the outlet points of these
    count = 0
    label = 1

    for id in sub_ids:
        size = np.count_nonzero(subs == id)
        size_df.loc[size_df.id == id, "size_km2"] = (size * 25 * 25) / 1e6

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


def get_giuh(subbasin_id, output_path, delta_t=15, plot=True):
    '''
    Classifies the traveltime raster and creates a unit hydrograph (time-area relationship).
    The hydrograph is written to file. THis function depends on more global variables which are not explicitly passed
    to the function. See traveltime.py.

    :param subbasin_id: int, id of subbasin
    :param output_path: str, path to store the output files
    :param delta_t: int, class size in minutes
    :param plot: plot (and save) the hydrographs
    :return:
    '''

    tt = rxr.open_rasterio(
        f'{output_path}traveltime/{subbasin_id}/tt_{subbasin_id}.tif')
    tt_vals = tt.values
    tt_vals_flat = tt_vals.flatten()
    tt_vals_flat = tt_vals_flat[~np.isnan(tt_vals_flat)]

    # cut of traveltimes larger than 5 days
    tt_vals_flat = tt_vals_flat[tt_vals_flat <= 60 * 24 * 5]

    max_bin = 60 * 24 * 5
    bins = np.arange(0, max_bin, delta_t)
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
    hydrograph = pd.DataFrame({'delta_t': bins, 'discharge_m3/s': count_list})
    hydrograph.to_csv(
        f'{output_path}{subbasin_id}/hydrograph_{subbasin_id}.csv')

    if plot:

        if not os.path.exists(f'{output_path}giuh_plots'):
            os.mkdir(f'{output_path}giuh_plots')

        outlet_df = pd.read_csv(
            f'{output_path}{subbasin_id}/outlet_{subbasin_id}.csv')
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
        plt.savefig(f"{output_path}giuh_plots/{subbasin_id}_giuh.png")
        plt.close()


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
