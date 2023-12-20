import counterfactual_functions as cf
import geopandas as gpd
import pcraster as pcr
import os
import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
import pandas as pd


#tt_46 = rxr.open_rasterio("output/basins/46/tt_46.tif", mask_and_scale=True)
#cf.get_giuh(46)

#check if giuh_plot mit dem cluster raster auch so ein plot wird?
#Liegt es evt. nur an der Plotfunktion, die ich auf dem Cluster gar nicht genuitzt hab?

cf.hydrograph_and_cn_for_subbasins([46])


cluster = rxr.open_rasterio("input/version/18342/tt_18342.tif", mask_and_scale=True)
cluster = cluster.values[0, :, :]

local = tt_46.values[0, :, :]

dif = cluster - local
np.nanmin(dif)

outlet_df = pd.read_csv(f'output/basins/46/outlet_46.csv')
x_coord = outlet_df.loc[0, "x"]
y_coord = outlet_df.loc[0, "y"]


import rioxarray as rxr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_giuh(tt_raster, subbasin_id, delta_t=15, plot=False):
    '''
    Same as ttfuncs.get_giuh just that it is declared here so it can use all the global variables without explictly
    :param subbasin_id:
    :param output_path:
    :param delta_t:
    :param plot:
    :return:
    '''

    tt_vals = tt_raster.values
    #tt_vals[tt_vals == tt_vals[0, 0, 0]] = np.nan
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
    hydrograph.to_csv(f'/home/voit/GIUH/hydrograph_{subbasin_id}.csv', index=False)

    x_coord = 0
    y_coord = 0

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
    plt.savefig(f"/home/voit/GIUH/{subbasin_id}_giuh.png")
    plt.close()


get_giuh(tt_46, 46)

id = 18342
clipped = rxr.open_rasterio(
    f'/work/voit/gis/main_basins/ger/output/basins/{id}/tt_{id}.tif',
    mask_and_scale=True)

id = 46
clipped = rxr.open_rasterio(f'/work/voit/gis/tt_46.tif', mask_and_scale=True)

get_giuh(clipped, id, plot=True)


get_giuh(tt_46, 46)

slopearea = pcr.readmap("/home/voit/GIUH/main_basins/ger/input/ger_slopearea.map")
basin_raster = pcr.readmap("/home/voit/GIUH/main_basins/ger/input/subbasins_adjusted.map")
basins_nominal = pcr.nominal(basin_raster)
    #get the average slopearea for each subbasin
mean_slopearea = pcr.areaaverage(slopearea, basins_nominal)

pcr.report(mean_slopearea, "/home/voit/GIUH/main_basins/ger/input/ger_mean_slopearea.map")

os.system(
    f'gdal_translate -of "PCRaster" -a_srs EPSG:3035 /home/voit/GIUH/paper_repo/counterfactual_flash_flood_analysis/input/mean_slope_area_cluster.tif /home/voit/GIUH/paper_repo/counterfactual_flash_flood_analysis/input/mean_slope_area_cluster.map'
)

input_path = "/home/voit/GIUH/main_basins/ger/input/"
flowdir = pcr.readmap(f'{input_path}ger_flowdir.map')
outlets = pcr.readmap(f'{input_path}outlets.map')
friction = pcr.readmap(
    f'{input_path}ger_friction_maidment.map')

outlets_vals = pcr.pcr2numpy(outlets, -99)

print(f'Number of outlets: {len(np.unique(outlets_vals))}')
position = outlets_vals > 0
position[position] = 1
position = position.astype(int)
outlet_raster = pcr.numpy2pcr(pcr.Boolean, position, -99)

print(f"Creating traveltime raster")
tt_complete = pcr.ldddist(flowdir, outlet_raster, friction)
tt_complete = tt_complete / 60
pcr.report(tt_complete, f'{input_path}generated/tt_complete.map')

