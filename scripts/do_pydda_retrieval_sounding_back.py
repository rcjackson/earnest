import pydda
import pyart
import xarray as xr
import numpy as np
import cmweather
import jax
import pandas as pd
import sys
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

grid_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/grids'
quicklook_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/quicklooks'

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
    parser = argparse.ArgumentParser(description='PyDDA dual-Doppler wind retrieval with sounding background')
    parser.add_argument('--Co', type=float, default=8.0,
                        help='Observation constraint weight (default: 8.0)')
    parser.add_argument('--Cm', type=float, default=1024.0,
                        help='Mass continuity constraint weight (default: 1024.0)')
    parser.add_argument('--Cb', type=float, default=1e-2,
                        help='Background constraint weight (default: 0.01)')
    parser.add_argument('--Cx', type=float, default=1000.0,
                        help='X-direction smoothness weight (default: 1000.0)')
    parser.add_argument('--Cy', type=float, default=1000.0,
                        help='Y-direction smoothness weight (default: 1000.0)')
    parser.add_argument('--Cz', type=float, default=1000.0,
                        help='Z-direction smoothness weight (default: 1000.0)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Convergence tolerance (default: 1e-3)')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--output_dir', type=str, default=grid_out_path,
                        help=f'Output directory for retrieved grids (default: {grid_out_path})')
    parser.add_argument('--quicklook_dir', type=str, default=quicklook_out_path,
                        help=f'Output directory for quicklook plots (default: {quicklook_out_path})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.quicklook_dir, exist_ok=True)

    ds_kbos = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/filter3x3/KBOX/KBOX20240213_180355_V06.nc')
    print("Loading KOKX data")

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
    print("Gridding KOKX")
    ds_kokx = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/KOKX/KOKX20240213_180223_V06.nc')
    grid_kokx = grid(ds_kokx)
    print("Initializing retrieval.")
    grid_kbox = pydda.io.read_from_pyart_grid(grid_kbox)
    grid_kokx = pydda.io.read_from_pyart_grid(grid_kokx)
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
    weights_kbox[~np.isfinite(weights_kbox)] = 0
    weights_kokx[~np.isfinite(weights_kokx)] = 0
    print("Loading sounding....")
    sounding = pd.read_csv('/projects/storm/rjackson/wfip3/soundings/okx_sounding_20240214.000000.csv')
    u_wind = sounding["SPED"] * -np.sin(np.deg2rad(sounding["DRCT"]))
    v_wind = sounding["SPED"] * -np.cos(np.deg2rad(sounding["DRCT"]))
    z = sounding["HGHT"]
    indicies = np.isfinite(u_wind.values)
    u_wind = u_wind[indicies]
    v_wind = v_wind[indicies]
    z = z[indicies]
    print(z.values)
    profile = pyart.core.HorizontalWindProfile.from_u_and_v(u_wind=u_wind, v_wind=v_wind, height=z)
    grid_kbox = pydda.initialization.make_wind_field_from_profile(grid_kbox, profile)
    bg_weights = 1 - weights_kbox
    bg_weights[0, :, :] = 0

    print(f"Running retrieval with Co={args.Co}, Cm={args.Cm}, Cb={args.Cb}, "
          f"Cx={args.Cx}, Cy={args.Cy}, Cz={args.Cz}, "
          f"tolerance={args.tolerance}, max_iterations={args.max_iterations}")

    grids, parameters = pydda.retrieval.get_dd_wind_field(
        [grid_kbox, grid_kokx],
        vel_name='corrected_velocity_unravel',
        Co=args.Co, Cm=args.Cm, engine="jax",
        u_back=u_wind.values, v_back=v_wind.values, z_back=z.values,
        Cb=args.Cb, tolerance=args.tolerance, max_iterations=args.max_iterations,
        Cx=args.Cx, Cy=args.Cy, Cz=args.Cz,
        weights_obs=[weights_kbox, weights_kokx], weights_bg=bg_weights)

    # Derive a timestamp string from the first grid's time coordinate
    time_val = pd.Timestamp(grids[0]["time"].values[0])
    time_str = time_val.strftime('%Y%m%d_%H%M%S')

    # Save retrieved grids
    for i, grid in enumerate(grids):
        del grid["time"].attrs["units"]
        out_file = os.path.join(args.output_dir, f"grid_{time_str}_{i}.nc")
        grid.to_netcdf(out_file)
        print(f"Saved grid to {out_file}")

    # Quicklook: wind barbs at levels 2, 5, and 8 (1-indexed -> 0-indexed: 1, 4, 7)
    plot_levels = [1, 4, 7]
    level_labels = ['Level 2', 'Level 5', 'Level 8']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, level, label in zip(axes, plot_levels, level_labels):
        pydda.vis.plot_horiz_xsection_barbs(
            grids,
            background_field='reflectivity',
            level=level,
            barb_spacing_x_km=10.0,
            barb_spacing_y_km=10.0,
            ax=ax)
        ax.set_title(f'{label} (~{grids[0]["z"].values[level]:.0f} m MSL)')

    fig.suptitle(f'Retrieved winds — {time_val.strftime("%Y-%m-%d %H:%M:%S UTC")}', fontsize=13)
    plt.tight_layout()

    quicklook_file = os.path.join(args.quicklook_dir, f"quicklook_{time_str}.png")
    fig.savefig(quicklook_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved quicklook to {quicklook_file}")
