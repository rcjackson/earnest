import pydda
import pyart
import xarray as xr
import numpy as np
import cmweather
import jax
import pandas as pd

def calculate_grid_points(grid_limits, resolution):
    nx = int((grid_limits[1] - grid_limits[0]) / resolution) + 1
    return nx

def grid(radar):
    print("Gatefilter start")
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid('corrected_velocity_region_based')
    print("Gatefilter made")
    grid = pyart.map.grid_from_radars([radar], grid_spec, (grid_z_range, grid_y_range, grid_x_range),
        gatefilters=[gatefilter],
        grid_origin=(origin_latitude, origin_longitude),
        grid_origin_alt=origin_altitude,
         weighting_function='Cressman')
    grid.fields["velocity"]["data"] = np.ma.masked_where(~np.isfinite(grid.fields["velocity"]["data"]),
                                                         grid.fields["velocity"]["data"])
    grid.fields["corrected_velocity_unravel"]["data"] = np.ma.masked_where(~np.isfinite(grid.fields["corrected_velocity_unravel"]["data"]),
                                                                           grid.fields["corrected_velocity_unravel"]["data"])
    grid.fields["corrected_velocity_region_based"]["data"] = np.ma.masked_where(~np.isfinite(grid.fields["corrected_velocity_region_based"]["data"]),
                                                                           grid.fields["corrected_velocity_region_based"]["data"])
    return grid
print("Loading KBOS data")

if __name__ == "__main__":
    ds_kbos = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/filter3x3/KBOX/KBOX20240213_180355_V06.nc')
    print("Loading KOKX data")
    #ds_kokx = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/KOKX/KOKX20240213_180223_V06.nc')

    grid_z_range = (0., 15000)
    grid_x_range = (-200000., 200000.)
    grid_y_range = (-200000., 200000.)
    grid_resolution_h = 1000.
    grid_resolution_v = 500.
    grid_spec = (calculate_grid_points(grid_z_range, grid_resolution_v),
             calculate_grid_points(grid_y_range, grid_resolution_h),
             calculate_grid_points(grid_x_range, grid_resolution_h))
    print(grid_spec)
    origin_latitude = ds_kbos.latitude["data"][0]
    origin_longitude = ds_kbos.longitude["data"][0]
    origin_altitude = ds_kbos.altitude["data"][0]
    print("Gridding data")
    grid_kbox = grid(ds_kbos)
    print("Griding kokx")
    ds_kokx = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/KOKX/KOKX20240213_180223_V06.nc')
    #grid_tbos = grid(ds_tbos)
    grid_kokx = grid(ds_kokx)
    print("Initalizing retrieval.")
    grid_kbox = pydda.io.read_from_pyart_grid(grid_kbox)
    #grid_tbos = pydda.io.read_from_pyart_grid(grid_tbos)
    grid_kokx = pydda.io.read_from_pyart_grid(grid_kokx)
    grid_kbox = pydda.initialization.make_constant_wind_field(grid_kbox, (0., 0., 0))
    bca0 = pydda.retrieval.get_bca(grid_kbox, grid_kokx)
    bca0 = np.tile(bca0, (grid_kbox.sizes["z"], 1, 1))
    bca = 30
    in_lobes = np.logical_and(bca0 >= np.deg2rad(bca), bca0 <= np.deg2rad(180 - bca))
    both_covered = np.logical_and(in_lobes, np.logical_and(np.isfinite(grid_kbox["corrected_velocity_unravel"].values).squeeze(),
                              np.isfinite(grid_kokx["corrected_velocity_unravel"].values)).squeeze())
    weights_kbox = np.logical_and(in_lobes, np.isfinite(grid_kbox["corrected_velocity_unravel"].values)).squeeze().astype(float)
    weights_kokx = np.logical_and(in_lobes, np.isfinite(grid_kokx["corrected_velocity_unravel"].values)).squeeze().astype(float)
    weights_kbox += both_covered.astype(float)
    weights_kokx += both_covered.astype(float)
    weights_kbox = weights_kbox/weights_kbox.max()
    weights_kokx = weights_kokx/weights_kokx.max()

    print("Loading sounding....")
    sounding = pd.read_csv('/projects/storm/wfip3/soundings/okx_sounding_20240214.000000.csv')
    u_wind = sounding["SPED"] * -np.sin(np.deg2rad(sounding["DRCT"])
    v_wind = sounding["SPED"] * -np.cos(np.deg2rad(sounding["DRCT"])
    z = sounding["HGHT"]
    grids = pydda.retrieval.get_dd_wind_field([grid_kbox, grid_kokx],
        vel_name='corrected_velocity_unravel', Co=1e-1, Cm=1024., engine="scipy",
        u_back=u_wind.data, v_back=v_wind.data, z_back = z.data, max_iterations=100,
                                              Cx=1e4, Cy=1e4, Cz=1e4, weights_obs=[weights_kbox, weights_kokx])

    i = 0
    for grid, parameters in grids:
        grid.to_netcdf(f"grid{i}.nc")
