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
import glob
import matplotlib
import pickle
from datetime import datetime, timedelta

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from herbie import Herbie

grid_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/grids'
quicklook_out_path = '/projects/storm/rjackson/wfip3/multidoppler_grids/quicklooks'
kbox_dir_default = '/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/filter3x3/KBOX'
kokx_dir_default = '/projects/storm/rjackson/wfip3/nexrad/nexrad_dealiased/KOKX'
sounding_dir_default = '/projects/storm/rjackson/wfip3/soundings'


def calculate_grid_points(grid_limits, resolution):
    nx = int((grid_limits[1] - grid_limits[0]) / resolution) + 1
    return nx


def calculate_average_kernel(size):
    return 1 / (size ** 3) * np.ones((1, size, size, size))


def find_nearest_radar_file(radar_dir, radar_name, target_time, tolerance_minutes=15):
    """Return the radar file whose timestamp is closest to target_time."""
    date_str = target_time.strftime('%Y%m%d')
    pattern = os.path.join(radar_dir, f'{radar_name}{date_str}_*_V06.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {radar_name} files found for {date_str} in {radar_dir}")

    best_file = None
    best_delta = timedelta(minutes=tolerance_minutes + 1)
    prefix_len = len(radar_name)
    for f in files:
        fname = os.path.basename(f)
        # filename format: KBOX20240213_180355_V06.nc
        time_part = fname[prefix_len:prefix_len + 15]  # YYYYMMDD_HHMMSS
        try:
            file_time = datetime.strptime(time_part, '%Y%m%d_%H%M%S')
        except ValueError:
            continue
        delta = abs(file_time - target_time)
        if delta < best_delta:
            best_delta = delta
            best_file = f

    if best_delta > timedelta(minutes=tolerance_minutes):
        raise FileNotFoundError(
            f"No {radar_name} file within {tolerance_minutes} min of {target_time}; "
            f"closest was {best_delta} away.")
    return best_file


def find_sounding_file(sounding_dir, target_time):
    """Return the sounding CSV closest in time for the target date."""
    date_str = target_time.strftime('%Y%m%d')
    pattern = os.path.join(sounding_dir, f'okx_sounding_{date_str}*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No sounding file found for {date_str} in {sounding_dir}")
    # Prefer the sounding whose time is closest to target_time
    best_file = None
    best_delta = timedelta.max
    for f in files:
        fname = os.path.basename(f)
        # expected format: okx_sounding_YYYYMMDD.HHMMSS.csv
        try:
            ts_part = fname.split('_')[-1].replace('.csv', '')  # YYYYMMDD.HHMMSS
            file_time = datetime.strptime(ts_part, '%Y%m%d.%H%M%S')
        except ValueError:
            continue
        delta = abs(file_time - target_time)
        if delta < best_delta:
            best_delta = delta
            best_file = f
    return best_file if best_file else files[0]


def make_grid(radar, origin_latitude, origin_longitude, origin_altitude,
              grid_z_range, grid_y_range, grid_x_range, grid_spec):
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid('velocity')
    g = pyart.map.grid_from_radars(
        [radar], grid_spec,
        (grid_z_range, grid_y_range, grid_x_range),
        gatefilters=[gatefilter],
        grid_origin=(origin_latitude, origin_longitude),
        grid_origin_alt=origin_altitude,
        roi_func='constant',
        constant_roi=1000.,
        weighting_function='Cressman')
    for field in ('velocity', 'corrected_velocity_unravel', 'corrected_velocity_region_based'):
        g.fields[field]['data'] = np.ma.masked_where(
            ~np.isfinite(g.fields[field]['data']),
            g.fields[field]['data'])
    return g


def process_time_step(target_time, args, grid_z_range, grid_y_range, grid_x_range, grid_spec):
    print(f"\n{'=' * 60}")
    print(f"Processing: {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 60}")

    # --- Radar files ---
    kbox_file = find_nearest_radar_file(args.kbox_dir, 'KBOX', target_time, args.radar_tolerance)
    kokx_file = find_nearest_radar_file(args.kokx_dir, 'KOKX', target_time, args.radar_tolerance)
    print(f"KBOX: {kbox_file}")
    print(f"KOKX: {kokx_file}")

    print("Loading and gridding KBOX")
    ds_kbos = pyart.io.read(kbox_file)
    origin_latitude = ds_kbos.latitude["data"][0]
    origin_longitude = ds_kbos.longitude["data"][0]
    origin_altitude = ds_kbos.altitude["data"][0]
    grid_kbox = make_grid(ds_kbos, origin_latitude, origin_longitude, origin_altitude,
                          grid_z_range, grid_y_range, grid_x_range, grid_spec)

    print("Loading and gridding KOKX")
    ds_kokx = pyart.io.read(kokx_file)
    print(f"  KOKX VCP: {ds_kokx.metadata['vcp_pattern']}")
    print(f"  KBOX VCP: {ds_kbos.metadata['vcp_pattern']}")
    grid_kokx = make_grid(ds_kokx, origin_latitude, origin_longitude, origin_altitude,
                          grid_z_range, grid_y_range, grid_x_range, grid_spec)

    # --- Convert to pydda grids ---
    print("Initializing retrieval")
    grid_kbox = pydda.io.read_from_pyart_grid(grid_kbox)
    grid_kokx = pydda.io.read_from_pyart_grid(grid_kokx)

    # --- BCA observation weights ---
    bca0 = pydda.retrieval.get_bca(grid_kbox, grid_kokx)
    bca0 = np.tile(bca0, (grid_kbox.sizes["z"], 1, 1))
    bca = 30
    in_lobes = np.logical_and(bca0 >= np.deg2rad(bca), bca0 <= np.deg2rad(180 - bca))
    both_covered = np.logical_and(
        in_lobes,
        np.logical_and(
            np.isfinite(grid_kbox["corrected_velocity_unravel"].values).squeeze(),
            np.isfinite(grid_kokx["corrected_velocity_unravel"].values).squeeze()))

    weights_kbox = np.isfinite(grid_kbox["corrected_velocity_unravel"].values).squeeze().astype(float)
    weights_kokx = np.isfinite(grid_kokx["corrected_velocity_unravel"].values).squeeze().astype(float)
    weights_kbox += both_covered.astype(float)
    weights_kokx += both_covered.astype(float)
    weights_kbox = weights_kbox / weights_kbox.max()
    weights_kokx = weights_kokx / weights_kokx.max()
    weights_kbox[~np.isfinite(weights_kbox)] = 0
    weights_kokx[~np.isfinite(weights_kokx)] = 0

    # Lower importance of winds in the purple haze ring (130–150 km)
    dist_kbox = np.sqrt(grid_kbox.point_x ** 2 + grid_kbox.point_y ** 2)
    phaze_kbox = np.logical_and(dist_kbox > args.phaze_min, dist_kbox < args.phaze_max)
    weights_kbox[phaze_kbox] /= args.phaze_factor

    # --- Sounding ---
    print("Loading sounding")
    sounding_file = find_sounding_file(args.sounding_dir, target_time)
    print(f"  Sounding: {sounding_file}")
    sounding = pd.read_csv(sounding_file)
    u_wind = sounding["SPED"] * -np.sin(np.deg2rad(sounding["DRCT"]))
    v_wind = sounding["SPED"] * -np.cos(np.deg2rad(sounding["DRCT"]))
    z = sounding["HGHT"]
    valid = np.isfinite(u_wind.values)
    u_wind = u_wind[valid]
    v_wind = v_wind[valid]
    z = z[valid]

    # --- IEM observations (cached per time step) ---
    time_str_cache = target_time.strftime('%Y%m%d_%H%M')
    iem_cache_file = f'iem_obs_{time_str_cache}.pkl'
    if not os.path.exists(iem_cache_file):
        iem_obs = pydda.constraints.get_iem_obs(grid_kbox)
        with open(iem_cache_file, 'wb') as p:
            pickle.dump(iem_obs, p)
    else:
        with open(iem_cache_file, 'rb') as p:
            iem_obs = pickle.load(p)

    # --- HRRR background ---
    # Round retrieval time down to the nearest hour for a valid HRRR output time,
    # then step back by fxx hours to get the run hour.
    hrrr_valid_time = target_time.replace(minute=0, second=0, microsecond=0)
    hrrr_run_time = hrrr_valid_time - timedelta(hours=args.fxx)
    print(f"HRRR run: {hrrr_run_time.strftime('%Y-%m-%d %H:%M')} UTC  fxx={args.fxx} "
          f"(valid ~{hrrr_valid_time.strftime('%H:%M')} UTC)")
    H = Herbie(hrrr_run_time.strftime('%Y-%m-%d %H:%M'), model="hrrr", product="prs", fxx=args.fxx)
    H.download()

    grid_kbox = pydda.constraints.add_hrrr_constraint_to_grid(grid_kbox, H.grib)
    grid_kbox["u"] = grid_kbox["U_hrrr"]
    grid_kbox["v"] = grid_kbox["V_hrrr"]
    grid_kbox["w"] = grid_kbox["W_hrrr"]

    bg_weights = 1 - weights_kbox
    bg_weights[0, :, :] = 0

    print(f"Running retrieval: Co={args.Co} Cm={args.Cm} Cb={args.Cb} "
          f"Cx={args.Cx} Cy={args.Cy} Cz={args.Cz} "
          f"tol={args.tolerance} max_iter={args.max_iterations}")

    iterations = 0
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

    time_str = grids[0]["time"].dt.strftime('%Y%m%d_%H%M%S').values[0]
    grids[0]["spd"] = np.sqrt(grids[0]["u"] ** 2 + grids[0]["v"] ** 2)
    grids[0]["spd"].attrs["long_name"] = "Wind speed"
    grids[0]["spd"].attrs["units"] = 'm s-1'

    # --- Save grids ---
    for i, g in enumerate(grids):
        del g["time"].attrs["units"]
        out_file = os.path.join(
            args.output_dir,
            f"grid_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{i}.nc")
        g.to_netcdf(out_file)
        print(f"Saved grid: {out_file}")

    # --- Quicklook: reflectivity background ---
    plot_levels = [1, 2, 5]
    level_labels = ['0.5 km', '1 km', '3 km']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, level, label in zip(axes, plot_levels, level_labels):
        pydda.vis.plot_horiz_xsection_quiver(
            grids, background_field='reflectivity', level=level,
            quiver_spacing_x_km=30.0, quiver_spacing_y_km=30.0, ax=ax)
        ax.set_title(f'{label} (~{grids[0]["z"].values[level]:.0f} m MSL)')
        c = ax.contour(grids[0]["x"] * 1e-3, grids[0]["y"] * 1e-3,
                       grids[0]["spd"].isel(z=level, time=0).values,
                       levels=np.arange(0, 40, 5), cmap='Reds', linewidth=2)
        plt.clabel(c)
    fig.suptitle(f'Retrieved winds — {time_str} UTC', fontsize=13)
    plt.tight_layout()
    ql_file = os.path.join(
        args.quicklook_dir,
        f"quicklook_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{args.filter_iterations}.png")
    fig.savefig(ql_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved quicklook: {ql_file}")

    # --- Quicklook: wind speed background ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, level, label in zip(axes, plot_levels, level_labels):
        pydda.vis.plot_horiz_xsection_quiver(
            grids, background_field='spd', cmap='Spectral_r', vmin=0, vmax=30,
            level=level, quiver_spacing_x_km=30.0, quiver_spacing_y_km=30.0, ax=ax)
        ax.set_title(f'{label} (~{grids[0]["z"].values[level]:.0f} m MSL)')
        c = ax.contour(grids[0]["x"] * 1e-3, grids[0]["y"] * 1e-3,
                       grids[0]["spd"].isel(z=level, time=0).values,
                       levels=np.arange(0, 40, 5), cmap='Reds', linewidth=2)
        plt.clabel(c)
    fig.suptitle(f'Retrieved winds — {time_str} UTC', fontsize=13)
    plt.tight_layout()
    ql_file = os.path.join(
        args.quicklook_dir,
        f"quicklook_{time_str}_{args.Co}_{args.Cm}_{args.Cb}_{args.Cx}_{args.Cy}_{args.Cz}_{args.filter_iterations}_spd.png")
    fig.savefig(ql_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved quicklook: {ql_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyDDA dual-Doppler wind retrieval with sounding/HRRR background — serial time-period mode')

    # Time period
    parser.add_argument('--start_time', type=str, required=True,
                        help='Start of processing window, UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_time', type=str, required=True,
                        help='End of processing window, UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--time_step', type=int, default=10,
                        help='Minutes between retrieval targets (default: 10)')

    # HRRR
    parser.add_argument('--fxx', type=int, default=2,
                        help='HRRR forecast hour (default: 2). '
                             'Run hour = retrieval hour - fxx hours.')

    # Data paths
    parser.add_argument('--kbox_dir', type=str, default=kbox_dir_default,
                        help=f'Directory of KBOX radar files (default: {kbox_dir_default})')
    parser.add_argument('--kokx_dir', type=str, default=kokx_dir_default,
                        help=f'Directory of KOKX radar files (default: {kokx_dir_default})')
    parser.add_argument('--sounding_dir', type=str, default=sounding_dir_default,
                        help=f'Directory of sounding CSV files (default: {sounding_dir_default})')
    parser.add_argument('--radar_tolerance', type=int, default=15,
                        help='Max minutes between target time and radar file timestamp (default: 15)')

    # Retrieval weights
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
                        help='Point obs weight (default: 1000.0)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Convergence tolerance (default: 1e-3)')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum total iterations (default: 100)')
    parser.add_argument('--filter_iterations', type=int, default=10,
                        help='Iterations between low-pass filter applications (default: 10)')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Convolutional averaging kernel size (default: 3)')

    # Purple haze
    parser.add_argument('--phaze_min', type=float, default=130000,
                        help='Inner range of purple haze suppression (m, default: 130000)')
    parser.add_argument('--phaze_max', type=float, default=150000,
                        help='Outer range of purple haze suppression (m, default: 150000)')
    parser.add_argument('--phaze_factor', type=float, default=2,
                        help='Purple haze weight suppression factor (default: 2)')

    # Output
    parser.add_argument('--output_dir', type=str, default=grid_out_path,
                        help=f'Output directory for retrieved grids (default: {grid_out_path})')
    parser.add_argument('--quicklook_dir', type=str, default=quicklook_out_path,
                        help=f'Output directory for quicklook plots (default: {quicklook_out_path})')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.quicklook_dir, exist_ok=True)

    # Build time series
    start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')
    end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M')
    time_steps = []
    t = start_time
    while t <= end_time:
        time_steps.append(t)
        t += timedelta(minutes=args.time_step)

    print(f"Processing {len(time_steps)} time steps from {start_time} to {end_time} "
          f"(every {args.time_step} min, HRRR fxx={args.fxx})")

    # Grid geometry (constant across all time steps)
    grid_z_range = (0., 15000)
    grid_x_range = (-300000., 300000.)
    grid_y_range = (-300000., 300000.)
    grid_resolution_h = 1000.
    grid_resolution_v = 500.
    grid_spec = (
        calculate_grid_points(grid_z_range, grid_resolution_v),
        calculate_grid_points(grid_y_range, grid_resolution_h),
        calculate_grid_points(grid_x_range, grid_resolution_h))
    print(f"Grid spec (nz, ny, nx): {grid_spec}")

    failed = []
    for i, target_time in enumerate(time_steps):
        print(f"\n[{i + 1}/{len(time_steps)}]", end=" ")
        try:
            process_time_step(target_time, args, grid_z_range, grid_y_range, grid_x_range, grid_spec)
        except FileNotFoundError as e:
            print(f"SKIP — {e}")
            failed.append((target_time, str(e)))
        except Exception as e:
            print(f"ERROR at {target_time}: {e}")
            failed.append((target_time, str(e)))

    print(f"\nDone. {len(time_steps) - len(failed)}/{len(time_steps)} time steps completed.")
    if failed:
        print("Skipped/failed time steps:")
        for t, reason in failed:
            print(f"  {t.strftime('%Y-%m-%d %H:%M')}: {reason}")
