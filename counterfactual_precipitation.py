import yaml
import pandas as pd
import pyproj
import wradlib as wrl
import geopandas as gpd
from osgeo import osr
import xarray as xr
import numpy as np
import os
from shapely.geometry import box

#######################
# extract from RADKLIM
#######################
def move_event(event, new_centroid, original_centroid):
    '''
    :param event: xarray.DataArray
    :param new_centroid: shapely. point
    :param original_centroid: tuple(x,y)
    '''

    xs = float(new_centroid.x.iloc[0])
    ys = float(new_centroid.y.iloc[0])


    x = original_centroid[0]
    y = original_centroid[1]

    dx = x - xs
    dy = y - ys

    dummy = event
    dummy = dummy.assign_coords({
        'x': ('x', dummy['x'].values - dx, dummy.attrs),
        'y': ('y', dummy['y'].values - dy, dummy.x.attrs)
    })

    return dummy


def get_rainseries_from_radklim(subbasin_id, event_id, subbasins_radolanproj, original_event, original_centroid,
                                radolan_projection, utm_projection, stereo_projection):

    if os.path.exists(
            f"output/precipitation/event{event_id}_sub{subbasin_id}.csv"
    ):
        print(
            f"File output/precipitation/event{event_id}_sub{subbasin_id}.csv already exists"
        )
        return

    print(f"Starting event {event_id}, counterfactual centroid: sub {subbasin_id}")
    # move the precipiation grid to the centroid of one subbasin
    if subbasin_id == -999: #event in its original position
        shifted_event = original_event

    else:
        centroid = subbasins_radolanproj.loc[subbasins_radolanproj.sub_id == subbasin_id].centroid
        shifted_event = move_event(original_event, centroid, original_centroid)

    min_x = shifted_event.x.values.min()
    min_y = shifted_event.y.values.min()
    max_x = shifted_event.x.values.max()
    max_y = shifted_event.y.values.max()
    bounding_box = box(min_x, min_y, max_x, max_y)
    bounding_box = gpd.GeoDataFrame(geometry=[bounding_box], crs=radolan_projection)
    bounding_box = bounding_box.to_crs("EPSG:32632")

    '''
    read a subset of the subbasins of germany
    this step is not really needed for this small example but speeds up the process if one uses spatial domain,
    which is larger than the rainfall event
    '''
    basins = gpd.read_file("output/gis/subbasins_utm.gpkg",
                           mask=bounding_box)
    #select just the ones which are fully contained
    subset = gpd.sjoin(basins, bounding_box, how='inner', predicate='within')
    #for creating a dataframe later
    subset_ids = subset.sub_id.to_list()

    #detour writing a file and later deleting it, just to read it without problems into wradlib VectorSource
    subset.geometry.to_file(
        f"output/basins/{subbasin_id}/dummy_{subbasin_id}_{event_id}.gpkg"
    )
    subs = wrl.io.VectorSource(
        f"output/basins/{subbasin_id}/dummy_{subbasin_id}_{event_id}.gpkg",
        src_crs=utm_projection,
        trg_crs=utm_projection,
        name="trg")

    # Get RADOLAN grid coordinates - center coordinates
    x_rad, y_rad = np.meshgrid(shifted_event.x, shifted_event.y)
    grid_xy_radolan = np.stack([x_rad, y_rad], axis=-1)

    #clip the gridverts to the extend of the actual subbasins. This might speed it up a little, because then there
    # are now gridverts on the sides of Germany
    # Reproject the RADOLAN coordinates
    xy = wrl.georef.reproject(grid_xy_radolan,
                              src_crs=stereo_projection,
                              trg_crs=utm_projection)

    #make bounding box
    bbox = subs.extent
    buffer = 2000.0
    bbox = dict(
        left=bbox[0] - buffer,
        right=bbox[1] + buffer,
        bottom=bbox[2] - buffer,
        top=bbox[3] + buffer,
    )

    # assign as coordinates
    shifted_event = shifted_event.assign_coords({
        "xc": (
            ["y", "x"],
            xy[..., 0],
            dict(long_name="UTM Zone 32 Easting", units="m"),
        ),
        "yc": (
            ["y", "x"],
            xy[..., 1],
            dict(long_name="UTM Zone 32 Northing", units="m"),
        ),
    })
    shifted_event = shifted_event.where(
        (((shifted_event.yc > bbox["bottom"]) &
          (shifted_event.yc < bbox["top"]))
         & ((shifted_event.xc > bbox["left"]) &
            (shifted_event.xc < bbox["right"]))),
        drop=True,
    )

    # Create vertices for each grid cell
    # (MUST BE DONE IN NATIVE RADOLAN COORDINATES)
    grid_x, grid_y = np.meshgrid(shifted_event.x, shifted_event.y)
    gridres = original_event.x.diff("x")[0].values
    grdverts = wrl.zonalstats.grid_centers_to_vertices(grid_x, grid_y, gridres,
                                                       gridres)

    src = wrl.io.VectorSource(grdverts,
                              trg_crs=utm_projection,
                              name="src",
                              src_crs=stereo_projection)

    #   and computes the intersections
    zd = wrl.zonalstats.ZonalDataPoly(src, trg=subs, crs=utm_projection)

    # This object can actually compute the statistics
    obj = wrl.zonalstats.ZonalStatsPoly(zd)

    # We just call this object with any set of radar data and iterate through the layers of a ncdf
    dummy = shifted_event.isel(time=0)
    avg = obj.mean(dummy.values.ravel())
    avg = np.reshape(avg, (1, -1))

    for hour in range(1, shifted_event.shape[0]):
        dummy = shifted_event.isel(time=hour)
        avg_new = obj.mean(dummy.values.ravel())
        avg_new = np.reshape(avg_new, (1, -1))
        avg = np.append(avg, avg_new, axis=0)


    #maybe one could just write the subbasins which recieved rainfall > 0
    df = pd.DataFrame(data=avg, columns=subset_ids)
    df = df.round(1)
    df.to_csv(
        f"output/precipitation/{event_id}_sub{subbasin_id}.gz",
        index=False,
        compression="gzip")

    os.remove(
        f"output/basins/{subbasin_id}/dummy_{subbasin_id}_{event_id}.gpkg"
    )


##########################
# effective precipitation
##########################

def calc_soil_moisture_class(row,  rain_df):
    '''
    function for pandas.apply which calculates the soil moisture based on the rainfall in the previous five days
    :param row: row of dataframe
    :return: int, soil moisture class for CN method
    '''

    if row.name - pd.Timedelta("5 days") in rain_df.index.to_list():
        rain_previous = rain_df.loc[row.name - pd.Timedelta("5 days"): row.name, :].sum(skipna=True, axis=0)

    else:
        rain_previous = rain_df.loc[rain_df.index[0]:row.name, :].sum(skipna=True, axis=0)

    sm_class = rain_previous.copy()
    sm_class.loc[rain_previous < 30] = 1
    sm_class.loc[(rain_previous >= 30) & (rain_previous < 50)] = 2
    sm_class.loc[rain_previous >= 50] = 3

    return sm_class


def get_effective_rainfall_scs(row, cn_table, rain_df):
    '''
    function for pandas.apply. Calculates the effective rainfall based on the curve number method
    :param row: passed by pandas.apply
    :param f_cn: correction factor for CN values (when calibrating)
    :return:
    '''

    eff_rain = rain_df.loc[row.name, :]
    eff_rain.index = eff_rain.index.astype(int)

    row.index = row.index.astype(int)
    cn_table_subset = cn_table.loc[row.index, :]

    row[row == 1] = cn_table_subset.loc[row == 1, "cn_sm_1"]
    row[row == 2] = cn_table_subset.loc[row == 2, "cn_sm_2"]
    row[row == 3] = cn_table_subset.loc[row == 3, "cn_sm_3"]

    s_max = (25400 / row - 254) / 24
    initial_abstraction = (0.2 * s_max)

    eff_rain = eff_rain - initial_abstraction
    eff_rain.loc[eff_rain < 0] = 0
    d1 = eff_rain ** 2
    d2 = eff_rain + s_max
    eff_rain = round(d1 / d2, 1)

    return eff_rain


def get_effective_rainfall(sub_id):

    cn_table = pd.read_csv('output/gis/CN_subbasins_table.csv')
    cn_table.index = cn_table.sub_id

    print(f"Sub ID {sub_id}")

    rain = pd.read_csv(f'output/precipitation/nw_jul21_sub{sub_id}.gz',
                       compression="gzip")

    # kick out columns where there is just nan
    rain = rain.dropna(axis=1, how='all')

    # make a  dummy date index to easily calculate 5 days previous rain in CN method
    rain.index = pd.date_range(start='8/10/2023', periods=len(rain), freq="1h")

    sm_df = rain.apply(calc_soil_moisture_class, rain_df=rain, axis=1)
    effective_rain = sm_df.apply(get_effective_rainfall_scs, cn_table=cn_table, rain_df=rain, axis=1)

    effective_rain.to_csv(f'output/effective_rainfall/nw_jul21_sub{sub_id}.gz',
                          compression="gzip", index=False)
