import counterfactual_functions as cf
import geopandas as gpd
import pcraster as pcr
import os
import numpy as np

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


#clip traveltime raster on cluster
gdal_translate -projwin 4077275.0 3055550.0 4117050.0 3019225.0 -of GTiff /home/voit/GIUH/main_basins/ger/input/tt_complete.map /tmp/processing_apxfSC/f970c7c9cec94acaade3eb0d7b4ca1c0/OUTPUT.tif