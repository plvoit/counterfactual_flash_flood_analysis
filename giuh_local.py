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
    hydrograph.to_csv(
        f'output/basins/{subbasin_id}/hydrograph_{subbasin_id}.csv',
        index=False)

    if plot:
        if not os.path.exists('output/giuh_plots'):
            os.mkdir('output/giuh_plots')

        outlet_df = pd.read_csv(f'output/basins/{subbasin_id}/outlet_{subbasin_id}.csv')
        x_coord = outlet_df.loc[0, "x"]
        y_coord = outlet_df.loc[0, "y"]

        fig = plt.figure(figsize=(14, 8))
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