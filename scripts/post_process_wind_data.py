import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#from dask_jobqueue import PBSCluster
from distributed import Client, wait, LocalCluster

radar = ["KARX", "KDVN", "KFSD", "KOAX", "KDMX"]

derecho_list = ["20090618", "20100601", "20130612", "20120804", "20190720",
        "20100618", "20100718", "20110711", "20130624", "20170721", 
        "20140630", "20140701", "20180628", "20200810", "20211215",
        "20220705", "20230629", "20240524", "20240715"]

output_path = '/lcrc/group/earthscience/rjackson/Earnest/wind_product_b1/'

wind_data = {}

def radar_beam_height(range_km, elevation_deg, radar_height_m=0.0, k=4/3):
    """
    Calculate the radar beam height above ground level.

    Parameters:
        x_km (float or array): X from radar in km
        y_km (float or array): Y from radar in km
        elevation_deg (float): Elevation angle (in degrees).
        radar_height_m (float): Radar height above ground level (in meters).
        k (float): Refraction coefficient (default is 4/3 for standard atmosphere).

    Returns:
        float or np.ndarray: Beam height above ground level (in meters).
    """
    # Constants
    Re = 6371000.0  # Earth's radius in meters
    r = range_km # Convert km to meters
    theta = np.radians(elevation_deg)  # Convert degrees to radians

    # Beam height calculation
    height = np.sqrt(r**2 + (k * Re)**2 + 2 * r * k * Re * np.sin(theta)) - k * Re + radar_height_m
    return height

def between_two_values(var, x, y):
    return np.logical_and(var >= x, var <= y)

def correct_derecho_day(date, rad, tolerance=60):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    try:
        wind_data = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/Earnest/wind_product/%s/%s%s*.nc' %
                                  (rad, rad, date), mode="r", combine='nested', concat_dim='time')

    except OSError:
        return
    wind_data["range"] = np.sqrt(wind_data["x"]**2 + wind_data["y"]**2)
    wind_data["beam_height"] = (["x", "y"], radar_beam_height(wind_data["range"].values, 0.5))
    wind_data["correction_factor"] = (10 / wind_data["beam_height"])**0.14
        # Mask out cross-wind component and restrict to 10 km range
    wind_data["dir"] = np.rad2deg(np.arctan2(-wind_data["u"], -wind_data["v"]))
    wind_data["dir"] = xr.where(wind_data["dir"] < 0, wind_data["dir"] + 360,
                                         wind_data["dir"])
    wind_data["azimuth"] = np.rad2deg(np.arctan2(wind_data["x"], wind_data["y"]))
    wind_data["azimuth"] = xr.where(wind_data["azimuth"] < 0, wind_data["azimuth"] + 360,
                                     wind_data["azimuth"])
        
    mask = ~between_two_values(np.abs(
        wind_data["dir"] - wind_data["azimuth"]), 90-tolerance, 90+tolerance)
    mask = np.logical_and(mask,
            ~between_two_values(np.abs(
            wind_data["dir"] - wind_data["azimuth"]), 270-tolerance, 270+tolerance))
    wind_data["spd"] = wind_data["spd"].where(mask)
    wind_data.to_netcdf(os.path.join(output_path, f'EARNEST.wind.{date}.{rad}.b1.nc'))
    del wind_data

def main():
    #cluster = PBSCluster(cores=36, memory="128GB", 
    #        account="rainfall", walltime="2:00:00", processes=3)
    #cluster = cluster.scale(6)
    with Client(LocalCluster(n_workers=6)) as c: 
        for d in derecho_list:
            for r in radar:
                print(d, r)
                if not os.path.exists(os.path.join(output_path, f'EARNEST.wind.{d}.{r}.b1.nc')):
                    correct_derecho_day(d, r)

if __name__ == "__main__":
    main()

