"""
animate_wind_levels.py
----------------------
Animate retrieved multi-Doppler winds alongside the HRRR background winds at
user-selected vertical levels.

Reads the per-time-step grid files written by do_pydda_retrieval_sounding_back.py
(``grid_YYYYMMDD_HHMMSS_0.nc`` for KBOX with the ``U_hrrr``/``V_hrrr`` fields
attached). Each animation frame is a grid of panels: one row per requested
vertical level, two columns (retrieved | HRRR), with wind speed shaded and
quivers on top.

Example
-------
    python animate_wind_levels.py \
        --grid_dir /projects/storm/rjackson/wfip3/multidoppler_grids/grids \
        --levels 1 2 5 \
        --output retrieved_vs_hrrr.gif
"""

import argparse
import glob
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


GRID_FILENAME_RE = re.compile(r'grid_(\d{8}_\d{6})_(\d+)\.nc$')


def find_grid_files(grid_dir, radar_index, start_time, end_time):
    """Return chronologically sorted grid files matching the radar_index suffix.

    Filters to files whose embedded timestamp falls in [start_time, end_time]
    when those bounds are supplied.
    """
    pattern = os.path.join(grid_dir, f'grid_*_{radar_index}.nc')
    candidates = sorted(glob.glob(pattern))
    selected = []
    for f in candidates:
        m = GRID_FILENAME_RE.search(os.path.basename(f))
        if not m:
            continue
        ts = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        selected.append((ts, f))
    selected.sort()
    return [f for _, f in selected], [t for t, _ in selected]


def load_grid_levels(path, levels, hrrr_u_name, hrrr_v_name):
    """Read just the fields we plot, only at the levels we need.

    Returns a dict keyed by level with (u_retr, v_retr, spd_retr, u_hrrr,
    v_hrrr, spd_hrrr) NumPy arrays plus x/y coordinate arrays in km.
    """
    ds = xr.open_dataset(path)
    try:
        x_km = ds['x'].values * 1e-3
        y_km = ds['y'].values * 1e-3
        z_m = ds['z'].values

        out = {'x_km': x_km, 'y_km': y_km, 'z_m': z_m, 'levels': {}}
        for lvl in levels:
            u_r = np.asarray(ds['u'].isel(z=lvl, time=0).values, dtype=float)
            v_r = np.asarray(ds['v'].isel(z=lvl, time=0).values, dtype=float)
            u_h = np.asarray(ds[hrrr_u_name].isel(z=lvl, time=0).values, dtype=float)
            v_h = np.asarray(ds[hrrr_v_name].isel(z=lvl, time=0).values, dtype=float)
            spd_r = np.sqrt(u_r ** 2 + v_r ** 2)
            spd_h = np.sqrt(u_h ** 2 + v_h ** 2)
            out['levels'][lvl] = dict(
                u_r=u_r, v_r=v_r, spd_r=spd_r,
                u_h=u_h, v_h=v_h, spd_h=spd_h)
        return out
    finally:
        ds.close()


def quiver_stride(n_x, n_y, spacing_km, dx_km, dy_km):
    """Pick a stride so quivers are roughly spacing_km apart on the plot."""
    sx = max(1, int(round(spacing_km / max(dx_km, 1e-6))))
    sy = max(1, int(round(spacing_km / max(dy_km, 1e-6))))
    return sx, sy


def draw_frame(fig, axes, cbar_axes, grid_data, levels, vmin, vmax, cmap,
               quiver_spacing_km, quiver_scale, title):
    """Render one frame in-place on the provided axes."""
    x_km = grid_data['x_km']
    y_km = grid_data['y_km']
    z_m = grid_data['z_m']
    dx_km = x_km[1] - x_km[0]
    dy_km = y_km[1] - y_km[0]
    sx, sy = quiver_stride(len(x_km), len(y_km), quiver_spacing_km, dx_km, dy_km)
    X, Y = np.meshgrid(x_km, y_km)

    last_mesh = None
    for row, lvl in enumerate(levels):
        d = grid_data['levels'][lvl]
        for col, (u, v, spd, label) in enumerate((
                (d['u_r'], d['v_r'], d['spd_r'], 'Retrieved'),
                (d['u_h'], d['v_h'], d['spd_h'], 'HRRR'))):
            ax = axes[row, col]
            ax.clear()
            mesh = ax.pcolormesh(
                x_km, y_km, spd, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.quiver(
                X[::sy, ::sx], Y[::sy, ::sx],
                u[::sy, ::sx], v[::sy, ::sx],
                scale=quiver_scale, color='black', width=0.0025, pivot='middle')
            ax.set_aspect('equal')
            ax.set_xlim(x_km[0], x_km[-1])
            ax.set_ylim(y_km[0], y_km[-1])
            ax.set_title(f'{label} — z={z_m[lvl]:.0f} m', fontsize=10)
            if row == len(levels) - 1:
                ax.set_xlabel('X [km]')
            if col == 0:
                ax.set_ylabel('Y [km]')
            last_mesh = mesh

    # One shared colorbar per row pair (or single colorbar on the right).
    if cbar_axes is not None and last_mesh is not None:
        cbar_axes.clear()
        fig.colorbar(last_mesh, cax=cbar_axes, label='Wind speed [m s$^{-1}$]')

    fig.suptitle(title, fontsize=13)


def build_animation(files, timestamps, levels, hrrr_u_name, hrrr_v_name,
                    vmin, vmax, cmap, quiver_spacing_km, quiver_scale,
                    figsize, dpi, fps, output):
    n_rows = len(levels)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=figsize, squeeze=False,
        gridspec_kw={'right': 0.88, 'wspace': 0.15, 'hspace': 0.3})
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.7])

    def update(frame):
        path = files[frame]
        ts = timestamps[frame]
        try:
            grid_data = load_grid_levels(path, levels, hrrr_u_name, hrrr_v_name)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [{frame + 1}/{len(files)}] SKIP {os.path.basename(path)}: {e}")
            return
        title = f'Retrieved vs HRRR winds — {ts.strftime("%Y-%m-%d %H:%M UTC")}'
        draw_frame(fig, axes, cbar_ax, grid_data, levels,
                   vmin, vmax, cmap, quiver_spacing_km, quiver_scale, title)
        print(f"  [{frame + 1}/{len(files)}] {ts.strftime('%Y-%m-%d %H:%M')}")

    anim = animation.FuncAnimation(
        fig, update, frames=len(files), repeat=False, blit=False)

    writer = animation.PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation: {output}")


def parse_optional_time(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M') if s else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Animate retrieved multi-Doppler winds vs HRRR background '
                    'at selected vertical levels.')
    parser.add_argument('--grid_dir', type=str, required=True,
                        help='Directory containing grid_*_0.nc files from '
                             'do_pydda_retrieval_sounding_back.py')
    parser.add_argument('--radar_index', type=int, default=0,
                        help='Grid file suffix to read (default: 0 = KBOX, '
                             'which carries the HRRR background fields)')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 5],
                        help='Z-indices to animate (default: 1 2 5)')
    parser.add_argument('--start_time', type=str, default=None,
                        help='Optional start time UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_time', type=str, default=None,
                        help='Optional end time UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--hrrr_u', type=str, default='U_hrrr',
                        help='HRRR U-wind variable name (default: U_hrrr)')
    parser.add_argument('--hrrr_v', type=str, default='V_hrrr',
                        help='HRRR V-wind variable name (default: V_hrrr)')
    parser.add_argument('--vmin', type=float, default=0.0,
                        help='Wind speed colormap minimum (default: 0)')
    parser.add_argument('--vmax', type=float, default=30.0,
                        help='Wind speed colormap maximum (default: 30)')
    parser.add_argument('--cmap', type=str, default='Spectral_r',
                        help='Background colormap (default: Spectral_r)')
    parser.add_argument('--quiver_spacing_km', type=float, default=30.0,
                        help='Approximate quiver spacing in km (default: 30)')
    parser.add_argument('--quiver_scale', type=float, default=400.0,
                        help='matplotlib quiver "scale" parameter — larger = '
                             'shorter arrows (default: 400)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second (default: 2)')
    parser.add_argument('--dpi', type=int, default=120,
                        help='Output DPI (default: 120)')
    parser.add_argument('--figsize', type=float, nargs=2, default=None,
                        help='Figure size W H in inches. Defaults to '
                             '(11, 4.5 * n_levels).')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GIF path')

    args = parser.parse_args()

    start_time = parse_optional_time(args.start_time)
    end_time = parse_optional_time(args.end_time)

    files, timestamps = find_grid_files(
        args.grid_dir, args.radar_index, start_time, end_time)
    if not files:
        raise SystemExit(
            f"No grid files matched in {args.grid_dir} "
            f"(radar_index={args.radar_index}, range={start_time}..{end_time})")
    print(f"Found {len(files)} grid file(s) in {args.grid_dir}")
    print(f"Levels: {args.levels}  Output: {args.output}")

    figsize = tuple(args.figsize) if args.figsize else (11.0, 4.5 * len(args.levels))

    build_animation(
        files=files,
        timestamps=timestamps,
        levels=args.levels,
        hrrr_u_name=args.hrrr_u,
        hrrr_v_name=args.hrrr_v,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        quiver_spacing_km=args.quiver_spacing_km,
        quiver_scale=args.quiver_scale,
        figsize=figsize,
        dpi=args.dpi,
        fps=args.fps,
        output=args.output,
    )
