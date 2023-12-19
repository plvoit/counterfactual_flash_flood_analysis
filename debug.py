import counterfactual_functions as cf
import geopandas as gpd
import pcraster as pcr
import os
import numpy as np
import rioxarray as rxr


tt_46 = rxr.open_rasterio("output/basins/46/tt_46.tif", mask_and_scale=True)
cf.get_giuh(tt_46, 46)

#check if giuh_plot mit dem cluster raster auch so ein plot wird?
#Liegt es evt. nur an der Plotfunktion, die ich auf dem Cluster gar nicht genuitzt hab?

cf.hydrograph_and_cn_for_subbasins([46])


cluster = rxr.open_rasterio("input/version/tt_46_cluster.tif", mask_and_scale=True)
test = rxr.open_rasterio("input/version/tt_46_cluster.tif", masked=True)
test = test.values[0, :, :]
cluster = cluster.values[0, :, :]
np.array_equal(cluster, test)

dif = cluster -test

local = tt_46.values[0, :, :]

dif = cluster - local
np.nanmin(dif)

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

