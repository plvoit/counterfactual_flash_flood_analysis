# Counterfactual_flash_flood_analysis

Supplementary model code for the manuscript:

**Voit, P. ; Heistermann, M.: A downward counterfactual analysis of flash floods in Germany**

https://nhess.copernicus.org/preprints/nhess-2023-224/

This repository contains the full workflow, derivation of hydrographs and the hydrological model for quick runoff.
DOI: 10.5281/zenodo.6556463

Contact: [voit@uni-potsdam.de](voit@uni-potsdam.de)

ORCIDs of first author:   
P. Voit:  [0000-0003-1005-0979](https://orcid.org/0000-0003-1005-0979)

For a detailed description please refer to the publication.

# Installation

The code was implemented in `Python 3.11`. Used packages are `wradlib` (Heistermann et al., 2013), `PCRaster` (Karssenberg
et al., 201), `GeoPandas` (Van den Bossche, 2023), `GDAL` (GDAL/OGR contributors, 2023), Numpy (Harris et al., 2020), `Pandas`
(McKinney, 2010), `Scikit-Learn` (Pedregosa et al., 2011), `Matplotlib` (Hunter, 2007),
`xarray` (Hoyer and Hamann, 2017). To view the notebooks you need to have `jupyter` installed.  
The user can use the included `counterfactual_env.yml` file to create a conda environment which includes all the necessary
packages. This can be done by:  
`conda config --add channels conda-forge`  
`conda config --set channel_priority strict`  
`conda env create -f counterfactual_env.yml`

Once the environment is installed it needs to be activated. Start the terminal of your choice and type:
`conda activate counterfactual`  
Then start Jupyter Notebook by typing:
`jupyter notebook`  
Now you can select and run the supplied notebooks in the browser.  
Alternatively you can manually install all the necessary packages without using a conda environment but this way it
can not be guaranteed that the packages work correctly.

If you have problems installing GDAL this might be of help:
https://gdal.org/api/python_bindings.html

# Included files
## Part 1 - Hydrographs_from_DEM.ipynb
This notebook demonstrates how the input files for the hydrological model and the hydrogprahs are created. 
As an examplary region, we use the region around Altenahr in West Germany. The function which are used in this 
notebook can be found in `counterfactual_preparation.py`.

## Part 2 - Rainfall input.ipynb
This notebooks shows how rainfall time series for the subbasins can be derived from RADKLIM rainfall estimates.
Secondly, the rainfall is transformed to effective precipitation, which serves as an input for the hydrological
model. The function which are used in this 
notebook can be found in `counterfactual_precipitation.py`.

## Part 3 - Modelling of quick runoff.ipynb
In this notebook we demonstrate how the counterfactual experiment was designed and run the hydrological model for quick
surface runoff. For every subbasin we create a counterfactual of the NW/Jul21 event and compare the results to
the original event. The function which are used in this  notebook can be found in `counterfactual_discharge_analysis.py`.

## counterfactual_preparation.py
Contains all the functions which are used to prepare the input (gis) files to run the model. Documentation is included.

## counterfactual_precipitation.py
Contains all the functions which are used to create rainfall time series for the subbasins. Documentation is included.

## counterfactual_discharge_analysis.py
Contains all the functions which are used to run the hydrological model. Documentation is included.

## counterfactual_env.yml
This file can be used to create a conda environment that includes all the necessary scripts to run the supplied
scripts. See "Installation".

# Included Data
All necesseray input files are in the folder `input`:

## buek250_cn_classes.gpkg
Vector data containing soil-landuse polygons and their associated curve numbers. Soil information was derived from
BUEK200 (BGR, 2018).

## cn_sm_classes.csv
Table containing curve numbers for soil moisture class 2 and their according values for soil moisture class 1 and 3.

## corine.gpkg
Vector data containing landuse information. Extracted from CORINE (BKG, 2018).

## dem.map
Digital elevation model extracted from EU-DEM (European Commission, 2016) in 25m x 25m resolution. All computed
rasters are based on this raster. The `.map`-format is specific for PCRaster but is included in the common GDAL format.
The GDAL library can be used to convert rasters to different datatypes (e.g. `-tif`to `.map`) with `gdal_translate`.

## NUTS5000_N1.*
Vector data containing the borders of the federal states of Germany. Just used for plotting.

## nw_jul21.nc
NetCDF compiled from RADKLIM (Winterrath et al., 2018) data with the library `radolan_to_netcdf` (Chwala, 2021). RADKLIM for the years 2001-today 
can be downloaded at the DWD open data server (https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/reproc/2017_002/).
Lately, the DWD also supplies RADKLIM in NetCDF but the code has not been tested with these files and will most likely need
to be adapted to different naming conventions etc.

## outlet_altenahr.gpkg
Vector data containing the outlet of the examplary catchemnt. Just used for plotting.

## scs.csv
Table containing CORINE landuse classes and their according curve numbers for for different soil classes (A, B, C, D).

## streams.gpkg
Vector data containing the river network in the examplary region. Just used for plotting.

# References
BGR: BÜK200 V5.5, BGR, https://www.bgr.bund.de/DE/Themen/Boden/Informationsgrundlagen/Bodenkundliche_Karten_Datenbanken/
BUEK200/buek200_node.html, 2018.

BKG: CORINE CLC5-2018, https://gdz.bkg.bund.de/index.php/default/open-data/corine-land-cover-5-ha-stand-2018-clc5-2018.html,

Chwala, C.: radolan_to_netcdf, GitHub [code], https://github.com/cchwala/radolan_to_netcdf (last access: 18 August 2022), 2021.

European Commission: Digital Elevation Model over Europe (EU-DEM), Eurostat, https://ec.europa.eu/eurostat/web/gisco/geodata/
reference-data/elevation/eu-dem/eu-dem-dd, 2016

GDAL/OGR contributors (2023). GDAL/OGR Geospatial Data Abstraction software Library. Open Source Geospatial Foundation. URL https://gdal.org 
DOI: 10.5281/zenodo.5884351

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

Heistermann, Maik, S. Jacobi, and T. Pfaff. "An open source library for processing weather radar data (wradlib)."
Hydrology and Earth System Sciences 17.2 (2013): 863-871.

Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: https://doi.org/10.5334/jors.148

Karssenberg, D., Schmitz, O., Salamon, P., de Jong, K., and Bierkens, M. F. P.: A software framework for construction
of process-based stochastic spatio-temporal models and data assimilation, Environmental Modelling & Software 25(4), 489-502, 2010. doi: 10.1016/j.envsoft.2009.10.004.

Data structures for statistical computing in python, McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010

Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825–2830.

Van den Bossche, Joris, et al. "geopandas/geopandas: v0. 13.2." Zenodo (2023).


Winterrath, T., Brendel, C., Hafer, M., Junghänel, T., Klameth, A., Lengfeld, K., Walawender, E., Weigl, E., and Becker, A.: RAD-
KLIM Version 2017.002: Reprocessed gauge-adjusted radar data, one-hour precipitation sums (RW), Deutscher Wetterdienst (DWD),
https://doi.org/10.5676/DWD/RADKLIM_RW_V2017.002, 2018.