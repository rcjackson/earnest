import sys

import herbie
import xarray as xr
import metpy
import fsspec
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import zarr
import pygrib
import os

from scipy.spatial import KDTree

lat_lons = {'KARX': (43.820833, -91.191111),
            'KOAX': (41.320369, -96.366819),
            'KDVN': (41.611667, -90.580833),
            'KDMX': (41.731, 93.723),
            'KFSD': (43.587778, 96.729444)}

model = "hrrr"
date = sys.argv[1]
dses = {}
ds_pres = []
ds_surf = []
for i in range(24):
    print(i)
    if model == "rap_historical":
        file_path = f'/lcrc/group/earthscience/rjackson/Earnest/hrrr/{model}/{date}/ruc2anl_130_{date}_{i:02d}00_000.grb2'
        ds_pres.append(xr.open_dataset(file_path, engine="cfgrib", filter_by_keys={'typeOfLevel': 'isobaricInhPa'}))
        ds_surf.append(xr.open_dataset(file_path, engine="cfgrib", filter_by_keys={'typeOfLevel': 'surface'}))
    elif model == "hrrr":
        file_path_prs = f'/lcrc/group/earthscience/rjackson/Earnest/hrrr/{model}/{date}/hrrr.t{i:02d}z.wrfprsf00.grib2'
        file_path_surf = f'/lcrc/group/earthscience/rjackson/Earnest/hrrr/{model}/{date}/hrrr.t{i:02d}z.wrfsfcf00.grib2'
        if os.path.exists(file_path_prs) and os.path.exists(file_path_surf):
            u = xr.open_dataset(file_path_prs, engine="cfgrib", filter_by_keys={'typeOfLevel': 'isobaricInhPa', "shortName": "u", "level": 850})
            v = xr.open_dataset(file_path_prs, engine="cfgrib", filter_by_keys={'typeOfLevel': 'isobaricInhPa', "shortName": "v", "level": 850})
            ds = xr.merge([u, v])
            print(ds)
            ds_pres.append(ds)
            ds = xr.open_dataset(file_path_surf, engine="cfgrib", filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'})
            ds = ds[["cape", "time"]]
            ds_surf.append(ds)
        print(i)

ds_pres = xr.concat(ds_pres, dim='time')
ds_surf = xr.concat(ds_surf, dim='time')
print(ds_pres)
print(ds_surf)
tree = KDTree(np.c_[ds_pres["latitude"].values.ravel(), ds_pres["longitude"].values.ravel()])
u850 = ds_pres["u"]
v850 = ds_pres["v"]
spd850 = np.sqrt(u850**2 + v850**2)
if model == "rap_historical":
    cape = ds_surf["unknown"]
    thetae = ds_surf["p3014"]
elif model == "hrrr":
    cape = ds_surf["cape"]
datasets = {}
for k in lat_lons.keys():
    datasets[k] = {}
    index = int(tree.query((lat_lons[k][0], lat_lons[k][1]))[0])
    shape = (u850.shape[0], u850.shape[1]*u850.shape[2])
    datasets[k]['u850'] = (['time'], u850.data.reshape(shape)[:, index])
    datasets[k]['v850'] = (['time'], v850.data.reshape(shape)[:, index])
    datasets[k]['spd850'] = (['time'], spd850.data.reshape(shape)[:, index])
    datasets[k]['cape'] = (['time'], cape.data.reshape(shape)[:, index])
    datasets[k]['time'] = ds_pres["time"]
    datasets[k] = xr.Dataset(datasets[k])
    print(datasets[k])
    datasets[k].to_netcdf(f"{k}_{date}_met_params.nc")

