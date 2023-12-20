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
    hydrograph.to_csv(f'/home/voit/GIUH/hydrograph_{subbasin_id}.csv',
                      index=False)

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
