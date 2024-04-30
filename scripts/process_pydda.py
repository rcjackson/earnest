import pydda
import matplotlib.pyplot as plt
import cmweather
import xarray as xr

from herbie import Herbie

kdmx_grid = pydda.io.read_grid('/lcrc/group/earthscience/rjackson/Earnest/grids/KARX/KARX20120804_060253_V06.gz.nc')
H = Herbie(
    "2012-08-04T06:00:00",
    model="rap_historical",
    fxx=0,)
H.PRODUCTS
hrrr_ds_name = H.download()
hrrr_ds_name

kdmx_grid = pydda.constraints.add_hrrr_constraint_to_grid(kdmx_grid, hrrr_ds_name)
kdmx_grid["u"] = kdmx_grid["U_hrrr"]
kdmx_grid["v"] = kdmx_grid["V_hrrr"]
kdmx_grid["w"] = kdmx_grid["W_hrrr"]
iem_obs = pydda.constraints.get_iem_obs(kdmx_grid, window=60.)
iem_obs_ds_list = [xr.Dataset(x) for x in iem_obs]
iem_obs_ds = xr.concat(iem_obs_ds_list, dim='station')
iem_obs_ds.to_netcdf('test_iem.nc')
grids, _ = pydda.retrieval.get_dd_wind_field([kdmx_grid], Co=1.0, Cm=100.0, Cx=1e-2, Cy=1e-2,
                                            vel_name='corrected_velocity_region_based',
                                            engine='jax', points=iem_obs, Cpoint=1.0)

grids[0].to_netcdf('test_retrieval.nc')
