import matplotlib.pyplot as plt
import cmweather
import xarray as xr
import os
import glob
import pandas as pd
import sys
import numpy as np
import multiprocessing
import dask.bag as db

from datetime import datetime, timedelta
#from dask_cuda import LocalCUDACluster
from dask.distributed import Client, progress, LocalCluster
from herbie import Herbie
from scipy.signal import convolve2d

out_dir = '/lcrc/group/earthscience/rjackson/Earnest/wind_product/%s/' % sys.argv[1]
quicklook_dir = '/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/%s/' % sys.argv[1]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(quicklook_dir):
    os.makedirs(quicklook_dir)

date = sys.argv[2]
grid_dir = '/lcrc/group/earthscience/rjackson/Earnest/grids/%s/%s%s' % (sys.argv[1], sys.argv[1], date)
grid_list = sorted(glob.glob(grid_dir + '*.nc'))

radar = sys.argv[1]

if radar == "KDVN":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSFQw2jhMuftHYxVat_VphwwOqDEPsNnvQJ30wyyX13TsYgcx6671oY6EwVId8o7ay1t-GWmO7oY5lL/pub?output=csv')
elif radar == "KARX":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQLSMnks5De3ZXzfBTavGUgw6V4SRuLbdYwT-XmRLJwEwW3EAymHYpqIZV9FRGEdZe9qcNqjaU-HfR2/pub?output=csv')
elif radar == "KDMX":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSBvKeswWGD2OicJ6MVE4s1egiq6spyzgjkCeyoaxWLtNJyv8IHSn41-FhYB9E6NwDqUM98WT6EewEK/pub?output=csv')
elif radar == "KOAX":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR5naFO_jhIJ3yInwB8ZRsr5srb6gG4N6x-30mWUApOxCH9FTUiNh14cyEIOWBbzFBh3h2bZZdwRubj/pub?output=csv')
elif radar == "KFSD":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRtrZ7hwTruQc30peOg7S5qg6mHG0fjKQkQjmkYvzBtjZqyyaAmLk28PTnyPsu6_FoPZgYglHHBVYew/pub?output=csv')
elif radar == "KLOT":
    dealias_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1vQ-UAIP2EF8-Fr2XG-lbzTuisTZtDXlzcJYT15vB4YI/pub?output=csv')
    
def process_file(grid):
    import pydda
    base, name = os.path.split(grid)
    dealias_df_name = dealias_df[dealias_df["File"] == name]
    print(name)
    if len(dealias_df_name["Algorithm"].values) == 0:
        return
    wind_grid = pydda.io.read_grid(grid)
    time_str = wind_grid['time'].attrs["units"].split(" ")[-1]
    print(time_str)
    dt = datetime.fromisoformat(time_str)
    # Go to nearest hour, otherwise Herbie will round down
    if dt.minute > 30:
        dt = dt + timedelta(minutes=30)
    dt = datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)
    time_str = dt.isoformat()
    if dt.year > 2016:
        H = Herbie(
                time_str[:],
            model="hrrr",
            product="prs",
            fxx=0,)
        model = "RAP"
    else:
        H = Herbie(
                time_str[:],
            model="rap_historical",
            product="analysis",
            fxx=0,)
        model = "HRRR" 
    print(H.PRODUCTS)
    hrrr_ds_name = H.download()
    hrrr_ds_name

    wind_grid = pydda.constraints.add_hrrr_constraint_to_grid(wind_grid, hrrr_ds_name, method="nearest")
    wind_grid["u"] = wind_grid["U_hrrr"].fillna(0)
    wind_grid["v"] = wind_grid["V_hrrr"].fillna(0)
    wind_grid["w"] = wind_grid["W_hrrr"].fillna(0)
    
    wind_grid = wind_grid.isel(z=slice(0, 6))
#    iem_obs = pydda.constraints.get_iem_obs(wind_grid, window=60.)
#    for i in range(len(iem_obs)):
#        if np.isnan(iem_obs[i]['u']):
#           iem_obs.remove(iem_obs[i])
#    iem_obs_ds_list = [xr.Dataset(x) for x in iem_obs]
#    iem_obs_ds = xr.concat(iem_obs_ds_list, dim='station')
#    out_iem_name = os.path.join(grid_dir, name + '.iem.nc')
#    iem_obs_ds.to_netcdf()
#    print(dealias_df_name["Algorithm"].iloc[0])
    if dealias_df_name["Algorithm"].iloc[0] == 'r':
        vname = 'corrected_velocity_region_based'
    elif dealias_df_name["Algorithm"].iloc[0] == 'u':
        vname = 'corrected_velocity_unravel'
    else:
        wind_grid.close()
        return
    prev_spd = np.zeros_like(wind_grid["u"].values)
    cur_spd = np.sqrt(wind_grid["u"]**2 + wind_grid["v"]**2).values
    diff_speed = np.sqrt(
            wind_grid["U_hrrr"]**2 + wind_grid["V_hrrr"]**2) - np.abs(wind_grid[vname])
    weights = np.where(
        np.logical_and(np.isfinite(wind_grid[vname].values),
                                   diff_speed.values < 5), 1, 0)
    model_weights = np.where(weights == 1, 0, 1)
    i = 0
    while (np.abs(prev_spd - cur_spd)).max() > 0.9:
        prev_spd = np.sqrt(wind_grid["u"]**2 + wind_grid["v"]**2)
        if i > 1:
            kernel_size = 30
            kernel = 1 / kernel_size**2 * np.ones((kernel_size, kernel_size))
            u_convolved = np.zeros_like(wind_grid['u'])
            v_convolved = np.zeros_like(wind_grid['v'])
            w_convolved = np.zeros_like(wind_grid['w'])
            for i in range(wind_grid.sizes['z']):
                u_convolved[0, i] = convolve2d(wind_grid['u'].values[0, i], kernel, mode='same')
                v_convolved[0, i] = convolve2d(wind_grid['v'].values[0, i], kernel, mode='same')    
                w_convolved[0, i] = convolve2d(wind_grid['w'].values[0, i], kernel, mode='same')
            wind_grid['u'][:] = u_convolved
            wind_grid['v'][:] = v_convolved
            wind_grid['w'][:] = w_convolved
        grids, _ = pydda.retrieval.get_dd_wind_field([wind_grid], Cm=128.0, Co=1, vel_name=vname,
                                             wind_tol=0.1, Cx=0, Cy=0, Cz=0,
                                             Cmod=0, model_fields=["hrrr"], 
                                             max_iterations=50, engine='scipy',
                                             weights_obs=weights,
                                             weights_bg=model_weights,
                                             upper_bc=False)
        i = i + 1
        wind_grid = grids[0]
        cur_spd = np.sqrt(wind_grid["u"]**2 + wind_grid["v"]**2)
    out_grid_name = os.path.join(out_dir, name + '.radar.nc')
    grid = grids[0]
    grid['spd'] = np.sqrt(grid['u']**2 + grid['v']**2)
    grid['spd'].attrs["units"] = "m s-1"
    grid['spd'].attrs["long_name"] = "2D wind speed"
    grid.to_netcdf(out_grid_name)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    grid['spd'].isel(z=0).plot(cmap='coolwarm', vmin=0, vmax=60, ax=ax[0])
    pydda.vis.plot_horiz_xsection_quiver(grids, ax=ax[1], level=0)
    ax[1].set_title('Winds at 400 m')
    ax[0].set_ylabel('Y [m]')
    ax[0].set_xlabel('X [m]')
    ax[0].set_title(grid['time'].attrs["units"].split(" ")[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(quicklook_dir, name + '.wspd.png'), bbox_inches='tight', dpi=150)
    grid = grid.drop_vars(
            ["point_latitude", "point_longitude", "point_altitude", "AZ", "EL", "point_x", "point_y", "point_z"])
    grid.to_netcdf(out_grid_name)
    wind_grid.close()

def process_list(file_list):
    for fname in file_list:
        process_file(fname)

if __name__ == "__main__":
    i = 0
    process_list(grid_list)
