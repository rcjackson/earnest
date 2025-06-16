import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cmweather
import glob
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from dask_jobqueue import PBSCluster
from distributed import Client, wait

radar = ["KARX", "KDVN", "KFSD", "KOAX", "KDMX"]

derecho_list = ["20090618", "20100601", "20100718", "20110711", "20130624",
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
    r = range_km * 1000  # Convert km to meters
    theta = np.radians(elevation_deg)  # Convert degrees to radians

    # Beam height calculation
    height = np.sqrt(r**2 + (k * Re)**2 + 2 * r * k * Re * np.sin(theta)) - k * Re + radar_height_m
    return height

def between_two_values(var, x, y):
    return np.logical_and(var >= x, var <= y)

def correct_derecho_day(date, tolerance=60):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    for rad in radar:
        try:
            print(glob.glob('/lcrc/group/earthscience/rjackson/Earnest/wind_product/%s/%s%s*.nc' %
                                  (rad, rad, date)))
            wind_data[rad] = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/Earnest/wind_product/%s/%s%s*.nc' %
                                  (rad, rad, date), mode="r", parallel=True)
        except OSError:
            if rad in wind_data.keys():
                del wind_data[rad]
            continue
        wind_data[rad]["range"] = np.sqrt(wind_data[rad]["x"]**2 + wind_data[rad]["y"]**2)
        wind_data[rad]["beam_height"] = (["x", "y"], radar_beam_height(wind_data[rad]["range"].values, 0.5))
        wind_data[rad]["correction_factor"] = (10 / wind_data[rad]["beam_height"])**0.14
        # Mask out cross-wind component and restrict to 10 km range
        wind_data[rad]["dir"] = np.rad2deg(np.arctan2(-wind_data[rad]["u"], -wind_data[rad]["v"]))
        wind_data[rad]["dir"] = xr.where(wind_data[rad]["dir"] < 0, wind_data[rad]["dir"] + 360,
                                         wind_data[rad]["dir"])
        wind_data[rad]["azimuth"] = np.rad2deg(np.arctan2(wind_data[rad]["x"], wind_data[rad]["y"]))
        wind_data[rad]["azimuth"] = xr.where(wind_data[rad]["azimuth"] < 0, wind_data[rad]["azimuth"] + 360,
                                     wind_data[rad]["azimuth"])
        
        mask = ~between_two_values(np.abs(
            wind_data[rad]["dir"] - wind_data[rad]["azimuth"]), 90-tolerance, 90+tolerance)
        mask = np.logical_and(mask,
            ~between_two_values(np.abs(
            wind_data[rad]["dir"] - wind_data[rad]["azimuth"]), 270-tolerance, 270+tolerance))
        wind_data[rad]["spd"] = wind_data[rad]["spd"].where(mask)
        wind_data[rad].to_netcdf(os.path.join(output_path, f'EARNEST.wind.{date}.{rad}.b1.nc'))
        del wind_data[rad]

def main():
    cluster = PBSCluster(cores=36, memory="128GB", 
            account="rainfall", walltime="2:00:00", processes=6)
    cluster = cluster.scale(6)
    with Client(cluster) as c: 
        results = c.map(correct_derecho_day, derecho_list)
        wait(results)

if __name__ == "__main__":
    main()

