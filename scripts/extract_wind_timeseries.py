import xarray as xr
import act
import pyart
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

station = sys.argv[2]
radar = sys.argv[1]
input_data_path = '/lcrc/group/earthscience/rjackson/Earnest/wind_product_b1/'
out_data_station = '/lcrc/group/earthscience/rjackson/Earnest/wind_product_station/'

file_list = sorted(glob.glob(os.path.join(input_data_path, f'EARNEST.wind.*.{radar}.b1.nc')))
dates = [x.split(".")[2] for x in file_list]
radar_winds = {}
asoses = {}
obs_winds = {}
for f in file_list:
    wind_data = xr.open_dataset(f)
    date = f.split(".")[2]
    inds = np.logical_and(wind_data["reflectivity"] > 0,
            np.abs(wind_data["corrected_velocity_region_based"]) > 2)
    wind_data["spd"] = wind_data["spd"].where(inds) * wind_data["correction_factor"]
    lon, lat = pyart.core.cartesian_to_geographic(
        wind_data["x"].values, wind_data["y"].values,
        dict(proj="pyart_aeqd", lat_0=wind_data["origin_latitude"].values[0].mean(),
        lon_0=wind_data["origin_longitude"].values[0].mean()))
    time_window = [datetime(wind_data.time.dt.year[0].values,
                            wind_data.time.dt.month[0].values,
                            wind_data.time.dt.day[0].values,
                            wind_data.time.dt.hour[0].values,
                            wind_data.time.dt.minute[0].values,
                            wind_data.time.dt.second[0].values),
                            datetime(wind_data.time.dt.year[-1].values,
                            wind_data.time.dt.month[-1].values,
                            wind_data.time.dt.day[-1].values,
                            wind_data.time.dt.hour[-1].values,
                            wind_data.time.dt.minute[-1].values,
                            wind_data.time.dt.second[-1].values)]
    asoses = act.discovery.get_asos_data(time_window, station=station)[station]
    asoses = asoses.drop_duplicates(dim='time', keep='first')
    lat_index = np.argmin(np.abs(asoses.attrs["site_latitude"] - lat))
    lon_index = np.argmin(np.abs(asoses.attrs["site_longitude"] - lon))
    radar_winds = wind_data["spd"].isel(x=lat_index, y=lon_index, z=0).load()
    obs_winds = asoses["spdms"].reindex(time=radar_winds.time, method='nearest').load()
    times = radar_winds.time.values
    dist = np.sqrt(wind_data["x"].isel(x=lat_index)**2 + wind_data["y"].isel(y=lat_index)**2).values
    print("%s processsed" % f)
    wind_data.close()
    ds_out = xr.Dataset({'time': (['time'], times),
        'radar_winds': (['time'], radar_winds.values),
        'obs_winds': (['time'], obs_winds.values)})
    ds_out["radar_winds"].attrs["long_name"] = f"Beam height-corrected radar estimated winds over {station}."
    ds_out["radar_winds"].attrs["units"] = "m/s"
    ds_out["obs_winds"].attrs["long_name"] = f"10 m measured winds over {station}."
    ds_out["obs_winds"].attrs["units"] = "m/s"
    ds_out.attrs["dist_drom_radar"] = dist
    ds_out.to_netcdf(os.path.join(out_data_station, f'EARNEST.timeseries.{date}.{station}.c1.nc'))
    del wind_data
