import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import xarray as xr
import os


class Event:
    '''
    This class handles all the geodata, hydrographs and rainfall time series to compute quick runoff and
    analyse the results. Additionally, there also some plotting functions
    '''
    def __init__(self,
                 event_id,
                 center_sub_id,
                 event_ncdf=None):

        '''
        Constructor for Event-Class
        :param event_id: str, the id of the rainfall event
        :param center_sub_id: id of the counterfactual/subbasin id where the rainfall event was moved to.
        :param event_ncdf: xarray.DataArray containing the RADKLIM rainfall estimates
        '''

        self.event_id = str(event_id)
        self.center_sub = center_sub_id

        if event_ncdf is None:
            self.ncdf = xr.open_dataset('input/nw_jul21.nc')
            self.event_start_time = self.ncdf.time[0].values

        else:
            self.ncdf = event_ncdf
            self.event_start_time = self.ncdf.time[0].values


        self.main_basin_shape = gpd.read_file(
            'output/gis/subbasins_info.gpkg'
        )
        self.main_basin_shape.rename(
            columns={"cum_upstream_area": "cum_upstre"}, inplace=True)

        self.floworder = pd.read_csv(
            'output/gis/flow_order.csv')
        self.floworder.sub_id = self.floworder.sub_id.apply(str)
        self.uncertainty_h = 3  # uncertainty assumed for the peak clash analysis

        self.eff_rain = pd.read_csv(
            f'output/effective_rainfall/{self.event_id}_sub{self.center_sub}.gz',
            compression='gzip')

        self.rain = pd.read_csv(
            f'output/precipitation/{self.event_id}_sub{self.center_sub}.gz',
            compression="gzip")

        print("Calculating discharge")
        self.discharge = self.direct_runoff_main_basin()


        self.eff_rain["date"] = pd.date_range(self.event_start_time,
                                              periods=len(self.eff_rain),
                                              freq="01 h")

        self.rain["date"] = pd.date_range(self.event_start_time,
                                          periods=len(self.rain),
                                          freq="01 h")

        self.dt = self.discharge.dt[0]  # dt of discharge/hydrograph
        self.discharge["date"] = pd.date_range(self.event_start_time,
                                               periods=len(self.discharge),
                                               freq="15 min")

        self.inflow_df = None
        self.inflow_df_for_subid = None
        self.inflow_unique_list = None

        self.ratio_df = self.peak_area_ratio_df()

        self.analysis_df = pd.merge(self.ratio_df,
                                    self.floworder[["sub_id", "order"]],
                                    on="sub_id")


        # add rain info to subbasin shape for plotting

        merge_list = [
            'sub_id', 'peak_m3/s', 'peak/area', 'total_rain',
            '1h_max_rain',
        ]
        self.analysis_df.sub_id = self.analysis_df.sub_id.astype(int)
        self.main_basin_shape = pd.merge(self.main_basin_shape,
                                         self.analysis_df[merge_list],
                                         left_on='sub_id',
                                         right_on='sub_id')

        # this bit is necessary for the Altenahr case study, because the inflow comes from two subcatchments
        if (("42" in self.discharge.columns) and ("37" in self.discharge.columns)):
            self.prepare_altenahr()

        self.analysis_df.rename(columns={"cum_upstre": "upstream_basin_area",
                                         "peak/area": "UPD"}, inplace=True)



    def plot_dis_and_rain(self, id):
        '''
        Plots the discharge and rainfall for a subbasin.
        :param id: int, subbasin id
        :return: plot
        '''

        id = str(id)

        if id == "altenahr":
            self.discharge["altenahr"] = self.discharge["37"] + self.discharge["42"]

        try:
            upstream_area = self.main_basin_shape.loc[
                self.main_basin_shape.index == int(id), "cum_upstre"].values[0]
        except:
            upstream_area = -999

        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()
        if not id == "altenahr":
            ax2.set_ylim((0, self.rain[id].max() * 2))
        ax2.invert_yaxis()
        ax2.tick_params(axis='y', colors="blue")
        ax1.plot(self.discharge["date"][:500],
                 self.discharge[id][:500],
                 label="runoff [m3/s]")
        if not id == "altenahr":
            ax2.bar(self.rain["date"][:500],
                    self.rain[id][:500],
                    label="precipitation [mm]",
                    width=0.1,
                    alpha=0.2)

        plt.title(f'{id}, upstream area: {upstream_area} km2')
        ax2.set_ylabel("precipitation [mm]", loc="top")
        ax1.set_ylabel("runoff [m3/s]")

        plt.show()
        plt.close()

    def recursive_upstream_search(self,
                                  id,
                                  inflow_basin_list,
                                  level_list,
                                  level=1):
        '''
        Called by get_upstream_basins.
        This function takes the floworder dataframe and uses its information to find all upstream basins for a given
        subbasin. It returns a dataframe containing the upstream basins and their order.
        Order means the starting subbasin (specified by id) has order one. Every basin that flows into this basin has order
        two. The more we move up the tree (level), the higher the order.
        :param id: int, id of basin from which we want to find all upstream basins
        :param inflow_basin_list: list of basins which flow into the current basins
        :param level_list: list of inflow basins levels (generated by other function)
        :param level: paramter needed for recursive function
        :return: returns a list of inflow basins and their levels (see function documentation of get_upstream_basins
        for more details)
        '''

        id = str(id)

        inflow_basins = self.floworder.loc[self.floworder.sub_id == id,
                                           "inflow_basins"].values

        if inflow_basins.size == 0:
            return

        inflow_basins = inflow_basins[0]

        level_list.append(level)

        if not isinstance(inflow_basins, str):
            return

        else:
            inflow_basins = inflow_basins.strip("[]").split(",")
            inflow_basin_list.extend(inflow_basins)

            for i in inflow_basins:
                i = i.strip(" ")
                self.recursive_upstream_search(i,
                                               inflow_basin_list,
                                               level_list,
                                               level=level + 1)


        return inflow_basin_list, level_list

    def get_upstream_basins(self, id):
        '''
        This function returns a dataframe which includes all the inflow basins in order from a given subbasin.
        This information helps to visualize the superposition of flows.
        flows from subbasins.
        :param id: subbasin from which the upstream inflow basins are found
        :param flow_order_df: dataframe derived by traveltime_functions.get_floworder
        :return: dataframe containing all the upstream basins in order. Recursion level describes the level up the tree:


                                            inflow_basin3       inflow_basin4   recursion level 3
                _____________________________________________|____________
                            inflow_basin1               inflow_basin2   recursion level 2
                _____________________|___________________________________
                                start basin         recursion level 1
        '''

        id = str(id)
        inflow_basin_list = []
        level_list = []
        upstream_basins, level_list = self.recursive_upstream_search(
            id, inflow_basin_list, level_list)

        upstream_basins = [i.strip(" ") for i in upstream_basins]
        upstream_basins.insert(0, id)

        inflow_df = pd.DataFrame({
            "inflow_basin": upstream_basins,
            "recursion_level": level_list
        })

        self.floworder.sub_id = self.floworder.sub_id.astype(str)
        inflow_df = pd.merge(inflow_df,
                             self.floworder[["sub_id", "order", "flows_to"]],
                             left_on="inflow_basin",
                             right_on="sub_id")
        inflow_df = inflow_df.sort_values("recursion_level", ascending=False)

        self.inflow_df = inflow_df
        self.inflow_df_for_subid = id

        # list of unique basins which contribute to inflow
        # because a set is not sorted anymore we have to do the following to keep the order and cant use a set
        unique_flow_to_list = []
        [
            unique_flow_to_list.append(x)
            for x in self.inflow_df.flows_to.to_list()
            if x not in unique_flow_to_list
        ]
        self.inflow_unique_list = unique_flow_to_list

        inflow_df.drop("inflow_basin", axis=1, inplace=True)

        return inflow_df


    def peak_area_ratio_df(self):
        '''
        Calculates the peak area ratio (Unit Peak Discharge) with a correction factor of 0.6
        :return: float, UPD
        '''

        maxima = self.discharge.drop(["dt", "date"], axis=1).max().round(1)

        mb = self.main_basin_shape.set_index("sub_id")

        ratio = pd.DataFrame({"peak_m3/s": maxima})
        ratio["sub_id"] = ratio.index.astype(int)
        ratio = pd.merge(ratio,
                         mb.cum_upstre.round(1),
                         left_on=ratio.sub_id,
                         right_on=mb.index)
        ratio["peak/area"] = round(ratio["peak_m3/s"] / ratio.cum_upstre**0.6,
                                   1)
        ratio = ratio.set_index("sub_id")
        ratio = ratio.rename(columns={"key_0": "id"})
        # add total rainfall, !not the effective rainfall
        rain_sum = pd.DataFrame(
            self.rain.drop("date", axis=1).sum(axis=0).round(1))
        rain_sum.index = rain_sum.index.astype(int)
        ratio = pd.merge(ratio,
                         rain_sum,
                         left_on=ratio.index,
                         right_on=rain_sum.index)
        ratio = ratio.rename(columns={0: "total_rain"})
        ratio.drop("key_0", axis=1, inplace=True)
        # maximum 1h rainfall during event
        rain_1hmax = pd.DataFrame(
            self.rain.drop("date", axis=1).max(axis=0).round(1))
        rain_1hmax.index = rain_1hmax.index.astype(int)
        ratio = pd.merge(ratio,
                         rain_1hmax,
                         left_on=ratio.id,
                         right_on=rain_1hmax.index)
        ratio = ratio.rename(columns={0: "1h_max_rain"})
        ratio.drop("key_0", axis=1, inplace=True)
        # ratio["marchi_envelope"] = round(97 * ratio.cum_upstre ** -0.4, 1)

        ratio["id"] = ratio["id"].astype(str)
        ratio.rename(columns={"id": "sub_id"}, inplace=True)

        return ratio


    def direct_runoff_main_basin(self, threshold=750):
        '''
        routes the rainfall through the main basin.
        Step 1: Superposition of rainfall for each subbasin and each subbasin.
        Step 2: Finding the flow order in the main basin. Head basins (order 1) don't depend on upstream inflow, hence they
        are caculated first. After that all the subbasins (order 2) which just have inflow from head basins are calculated. The
        inflow from the upstream head basins is added to the hydrograph of the order 2. We use the traveltime from the inlet
        of the order 2 basin to its outlet and add the upstream inflow after this traveltime to the hydrograph of the order
        2 basin. Then we iterate through all the orders. Result is a dataframe which stores all the discharge time series
        for all the subbasins of the main basin.

        :param threshold: size threshold of basins (km2) up to which discharge should be computed. For large basins the
        results of hydrograph based modelling are very uncertain.

        returns: writes a dataframe containing the discharge time series for every subbasin.
        '''

        # missing values in rain mess up all the superposition, so I set them to zero:
        rain = self.eff_rain.fillna(0)

        filter_list = self.main_basin_shape.loc[
            self.main_basin_shape.cum_upstre <= threshold, "sub_id"]
        filter_list = [str(i) for i in filter_list]
        # filter rain to just basins smaller than threshold
        rain = rain.loc[:, rain.columns.isin(filter_list)]

        col_names = list(rain.columns)
        col_names.insert(0, "dt")
        arr = np.zeros(((3000, len(col_names))))

        # just to get the right dt
        dt_val = 15
        dis_df = pd.DataFrame(columns=col_names, data=arr)
        dt = list(range(dt_val, 1000000, dt_val))
        dis_df.dt = dt[:len(dis_df)]

        # head catchments
        for sub_id in col_names[1:]:

            count_increment = int(60 / dt_val)
            uh = pd.read_csv(
                f'output/basins/{sub_id}/hydrograph_{sub_id}.csv')
            '''
            some unit hydrographs are really long. Usually they get weird, when there is a lake involved, e.g. subbasin
            3194.
            To keep the size reasonable we got the hydrograph after a length of 5 days, as we are also only interested in 
            flash floods with short response times and most events from the top 10 don't last longer than 5 days
            '''
            if len(uh) > 480:
                uh = uh.loc[:480, :]


            eff_rain_convolve = rain.loc[:, sub_id].values / 4
            a = np.convolve(np.repeat(eff_rain_convolve, 4), uh.loc[:, "discharge_m3/s"].to_numpy())

            dis_df.loc[0:a.shape[0] - 1, str(sub_id)] = a

        levels = list(set(self.floworder.order.values))

        col_names.remove("dt")

        highest_level = self.floworder.loc[
            self.floworder.sub_id.isin(col_names), 'order'].max()

        for level in levels[1:int(highest_level + 1)]:

            dummy = self.floworder.loc[self.floworder.order == level, :]
            dummy = dummy.loc[dummy.sub_id.isin(col_names), :]

            # add inflow from head catchments (order 1) to catchments with order 2 and so on
            for sub_id in dummy.sub_id:

                inflow_basins = dummy.loc[dummy.sub_id == sub_id,
                                          "inflow_basins"].values
                inflow_basins = inflow_basins[0].strip("[]").split(",")
                inflow_basins = [int(i) for i in inflow_basins]

                if str(sub_id) not in dis_df.columns:
                    '''
                    Some basins receive no rain during the event, which is why they are not listed in the effective rainfall
                    dataframe. These basins might receive flow from upstream basins though, which is why they need to be
                    added here.
                    This solution below is more efficient even though it seems a bit awkward.
                    The way I did it before, threw performance warnings
                    '''
                    new_column = dis_df.iloc[:, 3].copy()
                    new_column.name = str(sub_id)
                    new_column[:] = 0
                    dis_df = pd.concat([dis_df, new_column], axis=1)

                for basin in inflow_basins:

                    if str(basin) in col_names:
                        add_at_position = self.floworder.loc[
                            self.floworder.sub_id == str(basin),
                            "time_to_next_outlet"].values[0]

                        # find out, at which dt to add
                        add_at_position = add_at_position // dt[0]
                        '''
                        now the inflow from the upstream basin gets added to the current subbasin
                        for this  we use the traveltime from the outlfow of the upstream basin to the outflow of the
                        current basin. This results in the variable add_at_position.
                        Because the dataframe has a given length, we have to figure out exactly where to add and not to 
                        create nan
                        We add all the discharge from upstream but we have to shorten this  series in the  end, so it
                        still fits into the dataframe
                        '''

                        # new version using pandas shift:
                        shifted_discharge = dis_df[str(basin)].shift(
                            add_at_position).fillna(0)
                        dis_df[str(
                            sub_id)] = dis_df[str(sub_id)] + shifted_discharge

        if not os.path.exists('output/discharge'):
            os.mkdir('output/discharge')

        if not os.path.exists(f'output/discharge/{self.event_id}'):
            os.mkdir(f'output/discharge/{self.event_id}')

        return dis_df.round(1)

    def prepare_altenahr(self):
        """
        This function is needed to prepare the results for Case Study 1 as in Altenahr two catchments flow together.
        :return: just adds a column to the discharge dataframe
        """

        self.discharge["altenahr"] = self.discharge["36"] + self.discharge["41"]
        cum_upstre = 728.6

        df = pd.DataFrame({"sub_id": -999,
                           "peak_m3/s": self.discharge["altenahr"].max(),
                           "cum_upstre": cum_upstre,
                           "peak/area": round(self.discharge["altenahr"].max() / cum_upstre**0.6, 1),
                           "total_rain": -99, #there is an extra script for this
                           "1h_max_rain": -99,
                           "order": 11,
                           }, index=[0])

        self.analysis_df = pd.concat([self.analysis_df, df], ignore_index=True)