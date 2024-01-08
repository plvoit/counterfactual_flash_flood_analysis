import os
import counterfactual_preparation as cp
import counterfactual_discharge_analysis as cda
import geopandas as gpd
import pyproj
from osgeo import osr
import wradlib as wrl
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rasterio.features import dataset_features
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from osgeo import gdal, ogr

input_raster_path = 'output/gis/subbasins_adjusted.map'
output_vector_path = 'output/gis/subbasins_test.gpkg' #Todo: change to "first"

# Open the raster dataset
input_raster = gdal.Open(input_raster_path)

# Create the output vector datasource
output_vector_driver = ogr.GetDriverByName('GPKG')
output_vector = output_vector_driver.CreateDataSource(output_vector_path)

# Create a new layer in the vector datasource
output_layer = output_vector.CreateLayer('subbasins_first', geom_type=ogr.wkbPolygon)
# Use gdal.Polygonize to convert raster to vector
gdal.Polygonize(input_raster.GetRasterBand(1), None, output_layer, -1, [], callback=None)

# Close the datasets
input_raster = None
output_vector = None


from osgeo_utils.gdal_polygonize import gdal_polygonize


status_code = gdal_polygonize(
    src_filename="output/gis/subbasins_adjusted.map",
    band_number=1,
    dst_filename="output/gis/subbasins_first.gpkg",
    driver_name="GPKG"
                )

#delete the polyogns stemming from missing values
subs = gpd.read_file(f'output/gis/subbasins_first.gpkg')
subs['geometry'] = subs.buffer(0)  # this fixes the geometries
subs = subs.loc[subs.DN != 0, :] #DN == 0 are the polygons stemming from missing values
subs = subs.dissolve(by="DN")  # join single part polygons with same ID
subs = subs.set_crs("epsg:3035", allow_override=True)
subs.to_file(f"output/gis/subbasins.gpkg")

basins = rxr.open_rasterio('output/gis/subbasins_adjusted.map', mask=True)
basins.rio.to_raster('output/gis/subbasins_dummy.tif')
#Or as a GeoDataFrame
with rio.open("output/gis/subbasins_dummy.tif") as ds:
    gdf = gpd.GeoDataFrame.from_features(dataset_features(ds, bidx=1, as_mask=True, geographic=False, band=False))
    gdf.to_file("testbasins.gpkg")

basins.plot()
plt.show()

subbasins = gpd.read_file('output/gis/subbasins_info.gpkg')
subbasins = subbasins.loc[subbasins.cum_upstream_area < 750, :]
subbasin_list = subbasins.sub_id.to_list()
subbasin_list.append(-999)
nw_jul21 = xr.open_dataset('input/nw_jul21.nc')
if not os.path.exists('output/analysis'):
    os.mkdir('output/analysis')

if not os.path.exists('output/analysis/nw_jul21'):
    os.mkdir('output/analysis/nw_jul21')

for id in subbasin_list:
    print(f'Modelled discharge for counterfactual {id}')
    event = cda.Event("nw_jul21", id, discharge_df=None, event_ncdf=nw_jul21)
    event.discharge.to_csv(f'output/discharge/nw_jul21/dis_sub{id}.gz', compression='gzip', index=False)
    event.analysis_df.to_csv(f'output/analysis/nw_jul21/analysis_sub{id}.gz', compression='gzip', index=False)


subs = gpd.read_file("output/gis/subbasins_info.gpkg")

for i in subs.sub_id:
    rain = pd.read_csv(f"output/precipitation/nw_jul21_sub{i}.gz")

    if len(rain.columns) != 57:
        print(f'{i} incomplete')


#are the rainseries complete? Was the whole area covered?



event_ncdf = xr.open_dataset("input/nw_jul21.nc")
event = cda.Event("nw_jul21", -999, discharge_df=None, event_ncdf=event_ncdf)
dis = event.discharge
dis.max()
# create the files for the projection information
radolanwkt = """PROJCS["Radolan Projection",
    GEOGCS["Radolan Coordinate System",
        DATUM["Radolan_Kugel",
            SPHEROID["Erdkugel",6370040,0]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]]],
    PROJECTION["Polar_Stereographic"],
    PARAMETER["latitude_of_origin",90],
    PARAMETER["central_meridian",10],
    PARAMETER["scale_factor",0.933012701892],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["kilometre",1,
        AUTHORITY["EPSG","9036"]],
    AXIS["Easting",SOUTH],
    AXIS["Northing",SOUTH]]
"""
radolanproj = pyproj.CRS.from_wkt(radolanwkt)
proj_stereo = wrl.georef.create_osr("dwd-radolan-sphere")

# This is our target projection (UTM)
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(32632)


event_id = "WG/Jul21"
subbasins = gpd.read_file("input/generated/subbasins_info.gpkg")
#project the subbasin shapefile to UTM
subbasins_utm = subbasins.to_crs("EPSG:32632")
subbasins_utm = subbasins_utm[["sub_id", "geometry"]]
subbasins_utm.to_file("input/generated/subbasins_utm.gpkg")

subbasins = subbasins.to_crs(radolanproj)

event = xr.open_dataset("input/nw_jul21.nc")
event = event.rainfall_amount

#get the coordinates of the centroid

t_max, y_max, x_max = np.argwhere(event.values == np.nanmax(event.values))[0]
centroid_x_coord = event.x[x_max].item()
centroid_y_coord = event.y[y_max].item()
original_centroid = (centroid_x_coord, centroid_y_coord)

if not os.path.exists("input/precipitation"):
    os.mkdir("input/precipitation")

cp.get_rainseries_from_radklim(subbasin_id=46,
                               event_id="nw_jul21",
                               subbasins_radolanproj=subbasins,
                               original_event=event,
                               original_centroid=original_centroid,
                               radolan_projection=radolanproj,
                               utm_projection=proj_utm,
                               stereo_projection=proj_stereo
                               )
