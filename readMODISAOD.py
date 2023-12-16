### 555b Final Project
### Script for reading and re-projecting MAIAC AOD files

### Where to download MAIAC AOD files: https://lpdaac.usgs.gov/products/mcd19a2v061/#tools (use NASA Earthdata Search)

### Reference material: https://drivendata.co/blog/predict-pm25-benchmark/ 
### https://github.com/singingsea/MODIS_onGit/blob/master/read_aod_and_calculate_pm25.py
### https://hdfeos.org/software/pyhdf.php 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC, SDS
import pyproj
from pyproj import CRS, Proj
from typing import Dict, List, Union
import geopandas as gpd
#from mpl_toolkits.basemap import Basemap
#from cloudpathlib import S3Path
# from pathlib import Path
# import random

# MAIAC files:
    
#filepath = r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\MODISdata"
FILE_NAME = r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\MODIS_WashOr_2021-07-18\MCD19A2.A2021199.h09v04.061.2023149031557.hdf"
granule_id = 'MCD19A2.A2021199.h09v04.061.2023149031557'

# Open file
hdf = SD(FILE_NAME,SDC.READ)

# List available SDS datasets
for dataset, metadata in hdf.datasets().items():
    dimensions, shape, _, _ = metadata
    print(f"{dataset}\n    Dimensions: {dimensions}\n    Shape: {shape}")

# Get raw AOD data
blue_band_AOD = hdf.select("Optical_Depth_055")
name, num_dim, shape, types, num_attr = blue_band_AOD.info()
calibration_dict = blue_band_AOD.attributes()

# Get metadata for re-projecting onto usable coords
list(hdf.attributes().keys())
raw_attr = hdf.attributes()["StructMetadata.0"]

# Construct grid metadata from text blob
group_1 = raw_attr.split("END_GROUP=GRID_1")[0]
hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])

# Parse expressions still wrapped in apostrophes
for key, val in hdf_metadata.items():
    try:
        hdf_metadata[key] = eval(val)
    except:
        pass

# Save attributes needed to align and reproject data
# Note that coordinates are provided in meters
alignment_dict = {
    "upper_left": hdf_metadata["UpperLeftPointMtrs"],
    "lower_right": hdf_metadata["LowerRightMtrs"],
    "crs": hdf_metadata["Projection"],
    "crs_params": hdf_metadata["ProjParams"]
}

# Loop over orbits to apply the attributes
def calibrate_data(dataset: SDS, shape: List[int], calibration_dict: Dict):
    """Given a MAIAC dataset and calibration parameters, return a masked
    array of calibrated data.
    
    Args:
        dataset (SDS): dataset in SDS format (e.g. blue band AOD).
        shape (List[int]): dataset shape as a list of [orbits, height, width].
        calibration_dict (Dict): dictionary containing, at a minimum,
            `valid_range` (list or tuple), `_FillValue` (int or float),
            `add_offset` (float), and `scale_factor` (float).
    
    Returns:
        corrected_AOD (np.ma.MaskedArray): masked array of calibrated data
            with a fill value of nan.
    """
    corrected_AOD = np.ma.empty(shape, dtype=np.double)
    for orbit in range(shape[0]):
        data = dataset[orbit, :, :].astype(np.double)
        invalid_condition = (
            (data < calibration_dict["valid_range"][0]) |
            (data > calibration_dict["valid_range"][1]) |
            (data == calibration_dict["_FillValue"])
        )
        data[invalid_condition] = np.nan
        data = (
            (data - calibration_dict["add_offset"]) *
            calibration_dict["scale_factor"]
        )
        data = np.ma.masked_array(data, np.isnan(data))
        corrected_AOD[orbit, : :] = data
    corrected_AOD.fill_value = np.nan
    return corrected_AOD

corrected_AOD = calibrate_data(blue_band_AOD, shape, calibration_dict)
pd.DataFrame(corrected_AOD.ravel(), columns=['AOD']).describe()

# Create meshgrid in same projection as sinusoidal projection granule's upper left and lower right coords
def create_meshgrid(alignment_dict: Dict, shape: List[int]):
    """Given an image shape, create a meshgrid of points
    between bounding coordinates.
    
    Args:
        alignment_dict (Dict): dictionary containing, at a minimum,
            `upper_left` (tuple), `lower_right` (tuple), `crs` (str),
            and `crs_params` (tuple).
        shape (List[int]): dataset shape as a list of
            [orbits, height, width].
    
    Returns:
        xv (np.array): x (longitude) coordinates.
        yv (np.array): y (latitude) coordinates.
    """
    # Determine grid bounds using two coordinates
    x0, y0 = alignment_dict["upper_left"]
    x1, y1 = alignment_dict["lower_right"]
    
    # Interpolate points between corners, inclusive of bounds
    x = np.linspace(x0, x1, shape[2], endpoint=True)
    y = np.linspace(y0, y1, shape[1], endpoint=True)
    
    # Return two 2D arrays representing X & Y coordinates of all points
    xv, yv = np.meshgrid(x, y)
    return xv, yv

xv, yv = create_meshgrid(alignment_dict, shape)

# Source: https://spatialreference.org/ref/sr-org/modis-sinusoidal/proj4js/
sinu_crs = Proj(f"+proj=sinu +R={alignment_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
wgs84_crs = CRS.from_epsg("4326")

def transform_arrays(
    xv: Union[np.array, float],
    yv: Union[np.array, float],
    crs_from: CRS,
    crs_to: CRS
):
    """Transform points or arrays from one CRS to another CRS.
    
    Args:
        xv (np.array or float): x (longitude) coordinates or value.
        yv (np.array or float): y (latitude) coordinates or value.
        crs_from (CRS): source coordinate reference system.
        crs_to (CRS): destination coordinate reference system.
    
    Returns:
        lon, lat (tuple): x coordinate(s), y coordinate(s)
    """
    transformer = pyproj.Transformer.from_crs(
        crs_from,
        crs_to,
        always_xy=True,
    )
    lon, lat = transformer.transform(xv, yv)
    return lon, lat

# Project sinu grid onto wgs84 grid
lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs)

def convert_array_to_df(
    corrected_arr: np.ma.MaskedArray,
    lat:np.ndarray,
    lon: np.ndarray,
    granule_id: str,
    crs: CRS,
    total_bounds: np.ndarray = None
):
    """Align data values with latitude and longitude coordinates
    and return a GeoDataFrame.
    
    Args:
        corrected_arr (np.ma.MaskedArray): data values for each pixel.
        lat (np.ndarray): latitude for each pixel.
        lon (np.ndarray): longitude for each pixel.
        granule_id (str): granule name.
        crs (CRS): coordinate reference system
        total_bounds (np.ndarray, optional): If provided,
            will filter out points that fall outside of these bounds.
            Composed of xmin, ymin, xmax, ymax.
    """
    lats = lat.ravel()
    lons = lon.ravel()
    n_orbits = len(corrected_arr)
    size = lats.size
    values = {
        "value": np.concatenate([d.data.ravel() for d in corrected_arr]),
        "lat": np.tile(lats, n_orbits),
        "lon": np.tile(lons, n_orbits),
        "orbit": np.arange(n_orbits).repeat(size),
        "granule_id": [granule_id] * size * n_orbits
        
    }
    
    df = pd.DataFrame(values).dropna()
    if total_bounds is not None:
        x_min, y_min, x_max, y_max = total_bounds
        df = df[df.lon.between(x_min, x_max) & df.lat.between(y_min, y_max)]
    
    gdf = gpd.GeoDataFrame(df)
    gdf["geometry"] = gpd.points_from_xy(gdf.lon, gdf.lat)
    gdf["lat"] = gdf.lat
    gdf["lon"] = gdf.lon
    gdf.crs = crs
    return gdf[["granule_id", "orbit", "geometry", "lat", "lon", "value"]].reset_index(drop=True)

gdf = convert_array_to_df(corrected_AOD, lat, lon, granule_id, wgs84_crs)
gdf.head(3)

def plot_gdf(gdf: gpd.GeoDataFrame, separate_bands: bool = True):
    """Plot the Point objects contained in a GeoDataFrame.
    Option to overlay bands.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with, at a minimum,
            columns for `orbit`, `geometry`, and `value`.
        separate_bands (bool): Plot each band on its own axis.
            Defaults to True.
    
    Displays a matplotlib scatterplot.
    """
    if separate_bands:
        num_orbits = gdf.orbit.max() + 1
        f, axes = plt.subplots(
            1,
            num_orbits,
            figsize=(20, 5),
            sharex=True,
            sharey=True
        )
        for i, ax in enumerate(axes):
            gdf_orbit = gdf[gdf.orbit == i]
            img = ax.scatter(
                x=gdf_orbit.geometry.x,
                y=gdf_orbit.geometry.y,
                c=gdf_orbit.value,
                s=0.1,
                alpha=1,
                cmap="RdYlBu_r"
            )
            ax.set_title(f"Band {i + 1}", fontsize=12)
    else:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        img = ax.scatter(
            x=gdf.geometry.x,
            y=gdf.geometry.y,
            c=gdf.value,
            s=0.15,
            alpha=1,
            cmap="RdYlBu_r"
        )
    f.colorbar(img)
    plt.suptitle("Blue Band AOD", fontsize=12)
    
# Plot each orbit individually
plot_gdf(gdf, separate_bands=True)

#Save dataframe with corrected AOD and corresponding lat/lons 
gdf.to_csv(r"C:\Users\minad\Documents\UBC\EOSC-555B\FinalProject\MAIACAOD.csv")

