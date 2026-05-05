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
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, gaussian_filter
from herbie import Herbie
grid_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/grids'
quicklook_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/quicklooks'

def calculate_grid_points(grid_limits, resolution):
    nx = int((grid_limits[1] - grid_limits[0]) / resolution) + 1
    return nx

def calculate_average_kernel(size):
    return 1/(size**3) * np.ones((1, size, size, size))

def grid(radar):
    print("Gatefilter start")
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid('velocity')
    print("Gatefilter made")
    grid = pyart.map.grid_from_radars([radar], grid_spec, (grid_z_range, grid_y_range, grid_x_range),
        gatefilters=[gatefilter],
        grid_origin=(origin_latitude, origin_longitude),
        grid_origin_alt=origin_altitude,
        roi_func='constant',
        constant_roi=1000.,
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
    parser.add_argument('--Cp', type=float, default=1000.0,
                        help='Point obs weight (default: 10.0)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Convergence tolerance (default: 1e-3)')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--output_dir', type=str, default=grid_out_path,
                        help=f'Output directory for retrieved grids (default: {grid_out_path})')
    parser.add_argument('--quicklook_dir', type=str, default=quicklook_out_path,
                        help=f'Output directory for quicklook plots (default: {quicklook_out_path})')
    parser.add_argument('--filter_iterations', type=int, default=10,
            help='Maximum number of iterations before loss-pass filter is applied')
    parser.add_argument('--kernel_size', type=int, default=3, 
            help='Size of convolutional averaging filter for low-pass filter.')
    parser.add_argument('--phaze_min', type=float, default=130000,
            help="Minimum range for purple haze wind suppression factor.")
    parser.add_argument('--phaze_max', type=float, default=150000,
            help="Maximum range for purple haze wind suppression factor.")
    parser.add_argument('--phaze_factor', type=float, default=2,
            help="Purple haze wind suppression factor.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.quicklook_dir, exist_ok=True)

    ds_kbos = pyart.io.read('/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/filter3x3/KBOX/KBOX20240213_180355_V06.nc')
    print("Loading KOKX data")

    grid_z_range = (0., 15000)
    grid_x_range = (-300000., 300000.)
    grid_y_range = (-300000., 300000.)
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
    print(ds_kokx.metadata["vcp_pattern"])
    print(ds_kbos.metadata["vcp_pattern"])
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
    weights_kbox = np.isfinite(grid_kbox["corrected_velocity_unravel"].values).squeeze().astype(float)
    weights_kokx = np.isfinite(grid_kokx["corrected_velocity_unravel"].values).squeeze().astype(float)
    weights_kbox += both_covered.astype(float)
    weights_kokx += both_covered.astype(float)
    weights_kbox = weights_kbox/weights_kbox.max()
    weights_kokx = weights_kokx/weights_kokx.max()
    weights_kbox[~np.isfinite(weights_kbox)] = 0
    weights_kokx[~np.isfinite(weights_kokx)] = 0

    # Lower the importance of winds 130-150 km from each radar (purple haze ring)
    dist_kbox = np.sqrt(grid_kbox.point_x**2 + grid_kbox.point_y**2)
    dist_kokx = np.sqrt(grid_kokx.point_x**2 + grid_kokx.point_y**2)
    phaze_kbox = np.logical_and(dist_kbox > args.phaze_min, dist_kbox < args.phaze_max)
    #phaze_kokx = np.logical_and(dist_kokx > args.phaze_min, dist_kokx < args.phaze_max)
    weights_kbox[phaze_kbox] /= args.phaze_factor
    #weights_kokx[phase_kokx] /= args.phaze_factor

    print("Loading sounding....")
    sounding = pd.read_csv('/projects/storm/rjackson/wfip3/soundings/okx_sounding_20240213.120000.csv')
    u_wind = sounding["SPED"] * -np.sin(np.deg2rad(sounding["DRCT"]))
    v_wind = sounding["SPED"] * -np.cos(np.deg2rad(sounding["DRCT"]))
    z = sounding["HGHT"]
    indicies = np.isfinite(u_wind.values)
    u_wind = u_wind[indicies]
    v_wind = v_wind[indicies]
    z = z[indicies]
    print(z.values)
    if not os.path.exists('iem_obs.pkl'):
        iem_obs = pydda.constraints.get_iem_obs(grid_kbox)
        with open('iem_obs.pkl', 'wb') as p:
            pickle.dump(iem_obs, p)
    else:
        with open('iem_obs.pkl', 'rb') as p:
            iem_obs = pickle.load(p)
    H = Herbie("2024-02-13 16:00", model="hrrr", product="prs", fxx=2)
    H.download()

    profile = pyart.core.HorizontalWindProfile.from_u_and_v(u_wind=u_wind, v_wind=v_wind, height=z)
    grid_kbox = pydda.constraints.add_hrrr_constraint_to_grid(grid_kbox, H.grib)

    #grid_kbox = pydda.initialization.make_initialization_from_iem_obs(
    #        grid_kbox, iem_obs, profile=profile)
    grid_kbox["u"] = grid_kbox["U_hrrr"]
    grid_kbox["v"] = grid_kbox["V_hrrr"]
    grid_kbox["w"] = grid_kbox["W_hrrr"]

    bg_weights = 1 - weights_kbox
    bg_weights[0, :, :] = 0

    print(f"Running retrieval with Co={args.Co}, Cm={args.Cm}, Cb={args.Cb}, "
          f"Cx={args.Cx}, Cy={args.Cy}, Cz={args.Cz}, "
          f"tolerance={args.tolerance}, max_iterations={args.max_iterations}")
    iterations = 0
    average_kernel = calculate_average_kernel(args.kernel_size)
    grids = [grid_kbox, grid_kokx]
    while iterations < args.max_iterations:
        grids, parameters = pydda.retrieval.get_dd_wind_field(
            grids,
            vel_name='corrected_velocity_unravel', 
            Co=args.Co, Cm=args.Cm, engine="jax", mask_outside_opt=False,
            u_back=u_wind.values, v_back=v_wind.values, z_back=z.values,
            model_fields=["hrrr"],
            Cmod=args.Cb, tolerance=args.tolerance, max_iterations=args.filter_iterations,
            Cx=args.Cx, Cy=args.Cy, Cz=args.Cz, low_pass_filter=False,
            weights_obs=[weights_kbox, weights_kokx], weights_model=[bg_weights])
        iterations += args.filter_iterations
        if iterations < args.max_iterations:
            grids[0]["u"][:] = gaussian_filter(grids[0]["u"].values, sigma=1.5)
            grids[0]["v"][:] = gaussian_filter(grids[0]["v"].values, sigma=1.5)
            grids[0]["w"][:] = gaussian_filter(grids[0]["w"].values, sigma=1.5)
    grids[0]["u"][:] = gaussian_filter(grids[0]["u"].fillna(0).values, sigma=1.5)
    grids[0]["v"][:] = gaussian_filter(grids[0]["v"].fillna(0).values, sigma=1.5)
    grids[0]["w"][:] = gaussian_filter(grids[0]["w"].fillna(0).values, sigma=1.5)
    # Derive a timestamp string from the first grid's time coordinate
    time_str = grids[0]["time"].dt.strftime('%Y%m%d_%H%M%S').values[0]
    grids[0]["spd"] = np.sqrt(grids[0]["u"]**2 + grids[0]["v"]**2)
    grids[0]["spd"].attrs["long_name"] = "Wind speed"
    grids[0]["spd"].attrs["units"] = 'm s-1'
    # Save retrieved grids
    for i, grid in enumerate(grids):
        del grid["time"].attrs["units"]
        out_file = os.path.join(args.output_dir, f"grid_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{i}.nc")
        grid.to_netcdf(out_file)
        print(f"Saved grid to {out_file}")

    # Quicklook: wind barbs at levels 2, 5, and 8 (1-indexed -> 0-indexed: 1, 4, 7)
    plot_levels = [1, 2, 5]
    level_labels = ['0.5 km', '1 km', '3 km']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, level, label in zip(axes, plot_levels, level_labels):
        pydda.vis.plot_horiz_xsection_quiver(
            grids,
            background_field='reflectivity',
            level=level,
            quiver_spacing_x_km=30.0,
            quiver_spacing_y_km=30.0,
            ax=ax)
        ax.set_title(f'{label} (~{grids[0]["z"].values[level]:.0f} m MSL)')
        c = ax.contour(grids[0]["x"]*1e-3, grids[0]["y"]*1e-3,
            grids[0]["spd"].isel(z=level, time=0).values, levels=np.arange(0, 40, 5), cmap='Reds', linewidth=2)
        plt.clabel(c)

    fig.suptitle(f'Retrieved winds — {time_str} UTC', fontsize=13)
    plt.tight_layout()

    quicklook_file = os.path.join(args.quicklook_dir, f"quicklook_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{args.filter_iterations}.png")
    fig.savefig(quicklook_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # Quicklook: wind barbs at levels 2, 5, and 8 (1-indexed -> 0-indexed: 1, 4, 7)
    plot_levels = [1, 2, 5]
    level_labels = ['0.5 km', '1 km', '3 km']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, level, label in zip(axes, plot_levels, level_labels):
        pydda.vis.plot_horiz_xsection_quiver(
            grids,
            background_field='spd',
            cmap='Spectral_r',
            vmin=0,
            vmax=30,
            level=level,
            quiver_spacing_x_km=30.0,
            quiver_spacing_y_km=30.0,
            ax=ax)
        ax.set_title(f'{label} (~{grids[0]["z"].values[level]:.0f} m MSL)')
        plt.clabel(c)

    fig.suptitle(f'Retrieved winds — {time_str} UTC', fontsize=13)
    plt.tight_layout()

    quicklook_file = os.path.join(args.quicklook_dir, f"quicklook_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{args.filter_iterations}_spd.png")
    fig.savefig(quicklook_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved quicklook to {quicklook_file}")
