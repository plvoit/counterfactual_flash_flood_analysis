import os

import counterfactual_preparation as cp
import geopandas as gpd
import pyproj
from osgeo import osr
import wradlib as wrl
import pandas as pd
import xarray as xr
import numpy as np


cp.create_basic_files("input/")

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
