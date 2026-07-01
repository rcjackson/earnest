"""
animate_reflectivity_quivers.py
-------------------------------
Animate the retrieved u/v wind quivers over combined radar reflectivity.

For each time step there is a pair of grid files written by the PyDDA
retrieval, one per radar suffix::

    grid_YYYYMMDD_HHMMSS_0.nc
    grid_YYYYMMDD_HHMMSS_1.nc

The background shading is the *combined* reflectivity: the element-wise
maximum of the ``reflectivity`` field from the ``_0`` and ``_1`` files
(``np.fmax``, so a valid value on either side wins over a missing one).
Wind quivers (``u``/``v`` from the ``_0`` file) are drawn only where the
combined reflectivity has a valid (non-NaN) value.

Example
-------
    python animate_reflectivity_quivers.py \
        --grid_dir /projects/storm/rjackson/wfip3/multidoppler_grids/grids \
        --level 1 \
        --output reflectivity_quivers.gif

Optionally overlay line contours of any of the u/v/w wind components — or of
the horizontal wind speed sqrt(u**2 + v**2) via ``speed`` — on top of the
reflectivity shading and quivers::

    python animate_reflectivity_quivers.py \
        --grid_dir /path/to/grids \
        --contour w --contour_levels -4 -2 2 4 \
        --output reflectivity_quivers_w.gif

    python animate_reflectivity_quivers.py \
        --grid_dir /path/to/grids \
        --contour speed --contour_levels 10 20 30 \
        --output reflectivity_quivers_speed.gif
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


def find_time_pairs(grid_dir, index_a, index_b, start_time, end_time):
    """Return chronologically sorted (timestamp, path_a, path_b) tuples.

    A time step is included only when both grid files (suffix ``index_a`` and
    ``index_b``) exist. Timestamps are filtered to [start_time, end_time] when
    those bounds are supplied.
    """
    pairs = {}
    for f in sorted(glob.glob(os.path.join(grid_dir, 'grid_*.nc'))):
        m = GRID_FILENAME_RE.search(os.path.basename(f))
        if not m:
            continue
        stamp, index = m.group(1), int(m.group(2))
        ts = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        pairs.setdefault(ts, {})[index] = f

    selected = []
    for ts in sorted(pairs):
        entry = pairs[ts]
        if index_a in entry and index_b in entry:
            selected.append((ts, entry[index_a], entry[index_b]))
        else:
            have = sorted(entry)
            print(f"  SKIP {ts:%Y-%m-%d %H:%M:%S}: need suffixes "
                  f"{index_a} & {index_b}, found {have}")
    timestamps = [t for t, _, _ in selected]
    files_a = [a for _, a, _ in selected]
    files_b = [b for _, _, b in selected]
    return timestamps, files_a, files_b


def load_frame(path_a, path_b, level, refl_name, u_name, v_name, w_name):
    """Load one frame's data.

    Returns x/y coordinate arrays (km), the z height (m), the combined
    reflectivity (NaN where invalid) and the u/v/w wind components — all as
    2-D NumPy arrays for the requested vertical ``level``. ``w`` is ``None``
    when the variable is absent from the ``_0`` grid file.
    """
    ds_a = xr.open_dataset(path_a)
    ds_b = xr.open_dataset(path_b)
    try:
        x_km = ds_a['x'].values * 1e-3
        y_km = ds_a['y'].values * 1e-3
        z_m = float(ds_a['z'].values[level])

        refl_a = np.asarray(ds_a[refl_name].isel(time=0, z=level).values, dtype=float)
        refl_b = np.asarray(ds_b[refl_name].isel(time=0, z=level).values, dtype=float)
        # Element-wise max; a valid value on either side beats a NaN.
        refl = np.fmax(refl_a, refl_b)

        u = np.asarray(ds_a[u_name].isel(time=0, z=level).values, dtype=float)
        v = np.asarray(ds_a[v_name].isel(time=0, z=level).values, dtype=float)
        w = None
        if w_name in ds_a:
            w = np.asarray(ds_a[w_name].isel(time=0, z=level).values, dtype=float)
        return dict(x_km=x_km, y_km=y_km, z_m=z_m, refl=refl, u=u, v=v, w=w)
    finally:
        ds_a.close()
        ds_b.close()


def quiver_stride(spacing_km, dx_km, dy_km):
    """Pick a stride so quivers are roughly spacing_km apart on the plot."""
    sx = max(1, int(round(spacing_km / max(abs(dx_km), 1e-6))))
    sy = max(1, int(round(spacing_km / max(abs(dy_km), 1e-6))))
    return sx, sy


# Per-component styling for wind contours (m/s).
CONTOUR_STYLE = {
    'u': dict(colors='#1f77b4', label='u [m/s]'),
    'v': dict(colors='#d62728', label='v [m/s]'),
    'w': dict(colors='#000000', label='w [m/s]'),
    'speed': dict(colors='#2ca02c', label='wind speed [m/s]'),
}


def contour_field(name, data):
    """Return the 2-D field to contour for ``name``.

    ``speed`` is the horizontal wind speed sqrt(u**2 + v**2), computed on the
    fly; all other names are looked up directly in ``data``.
    """
    if name == 'speed':
        u = data.get('u')
        v = data.get('v')
        if u is None or v is None:
            return None
        return np.hypot(u, v)
    return data.get(name)


def draw_contours(ax, x_km, y_km, data, contour_vars, contour_levels):
    """Overlay line contours of the requested wind components.

    Returns a list of (label, Line2D-proxy) legend handles for the components
    that were actually drawn.
    """
    from matplotlib.lines import Line2D

    handles = []
    for name in contour_vars:
        field = contour_field(name, data)
        if field is None or not np.any(np.isfinite(field)):
            continue
        style = CONTOUR_STYLE[name]
        cs = ax.contour(
            x_km, y_km, np.ma.masked_invalid(field),
            levels=contour_levels, colors=style['colors'],
            linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
        handles.append(Line2D([0], [0], color=style['colors'],
                              lw=1.5, label=style['label']))
    return handles


def draw_frame(fig, ax, cbar_ax, data, vmin, vmax, cmap,
               quiver_spacing_km, quiver_scale, title,
               contour_vars=(), contour_levels=None):
    """Render one frame in-place on the provided axes."""
    x_km = data['x_km']
    y_km = data['y_km']
    refl = data['refl']
    u = data['u']
    v = data['v']

    dx_km = x_km[1] - x_km[0]
    dy_km = y_km[1] - y_km[0]
    sx, sy = quiver_stride(quiver_spacing_km, dx_km, dy_km)
    X, Y = np.meshgrid(x_km, y_km)

    # Mask quivers to points with a valid combined reflectivity value.
    valid = np.isfinite(refl)
    sub = (slice(None, None, sy), slice(None, None, sx))
    qmask = valid[sub]

    ax.clear()
    mesh = ax.pcolormesh(
        x_km, y_km, np.ma.masked_invalid(refl),
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.quiver(
        X[sub][qmask], Y[sub][qmask],
        u[sub][qmask], v[sub][qmask],
        scale=quiver_scale, color='black', width=0.0022, pivot='middle')

    handles = draw_contours(ax, x_km, y_km, data, contour_vars, contour_levels)
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8,
                  framealpha=0.85)

    ax.set_aspect('equal')
    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(y_km[0], y_km[-1])
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_title(title, fontsize=12)

    if cbar_ax is not None:
        cbar_ax.clear()
        fig.colorbar(mesh, cax=cbar_ax, label='Reflectivity [dBZ]')


def build_animation(timestamps, files_a, files_b, level, refl_name, u_name,
                    v_name, w_name, vmin, vmax, cmap, quiver_spacing_km,
                    quiver_scale, contour_vars, contour_levels,
                    figsize, dpi, fps, output):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])

    def update(frame):
        ts = timestamps[frame]
        try:
            data = load_frame(files_a[frame], files_b[frame], level,
                              refl_name, u_name, v_name, w_name)
        except (KeyError, IndexError, FileNotFoundError) as e:
            print(f"  [{frame + 1}/{len(timestamps)}] SKIP "
                  f"{os.path.basename(files_a[frame])}: {e}")
            return
        title = (f'Combined reflectivity + retrieved winds — '
                 f'z={data["z_m"]:.0f} m — '
                 f'{ts:%Y-%m-%d %H:%M UTC}')
        draw_frame(fig, ax, cbar_ax, data, vmin, vmax, cmap,
                   quiver_spacing_km, quiver_scale, title,
                   contour_vars=contour_vars, contour_levels=contour_levels)
        print(f"  [{frame + 1}/{len(timestamps)}] {ts:%Y-%m-%d %H:%M}")

    anim = animation.FuncAnimation(
        fig, update, frames=len(timestamps), repeat=False, blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation: {output}")


def parse_optional_time(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M') if s else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Animate u/v wind quivers over the combined (max) '
                    'reflectivity from paired grid_*_0.nc / grid_*_1.nc files.')
    parser.add_argument('--grid_dir', type=str, required=True,
                        help='Directory containing grid_YYYYMMDD_HHMMSS_{0,1}.nc')
    parser.add_argument('--level', type=int, default=1,
                        help='Z-index to plot (default: 1)')
    parser.add_argument('--index_a', type=int, default=0,
                        help='First grid suffix; also the source of u/v '
                             '(default: 0)')
    parser.add_argument('--index_b', type=int, default=1,
                        help='Second grid suffix for the reflectivity max '
                             '(default: 1)')
    parser.add_argument('--start_time', type=str, default=None,
                        help='Optional start time UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_time', type=str, default=None,
                        help='Optional end time UTC (YYYY-MM-DD HH:MM)')
    parser.add_argument('--refl_var', type=str, default='reflectivity',
                        help='Reflectivity variable name (default: reflectivity)')
    parser.add_argument('--u_var', type=str, default='u',
                        help='U-wind variable name (default: u)')
    parser.add_argument('--v_var', type=str, default='v',
                        help='V-wind variable name (default: v)')
    parser.add_argument('--w_var', type=str, default='w',
                        help='W-wind (vertical) variable name (default: w)')
    parser.add_argument('--contour', nargs='*',
                        choices=['u', 'v', 'w', 'speed'],
                        default=[], metavar='VAR',
                        help='Wind component(s) to overlay as line contours; '
                             'any of u v w speed, where "speed" is the '
                             'horizontal wind speed sqrt(u^2+v^2) (default: none)')
    parser.add_argument('--contour_levels', nargs='*', type=float, default=None,
                        help='Explicit contour levels in m/s (default: auto-'
                             'chosen by matplotlib)')
    parser.add_argument('--vmin', type=float, default=-10.0,
                        help='Reflectivity colormap minimum dBZ (default: -10)')
    parser.add_argument('--vmax', type=float, default=40.0,
                        help='Reflectivity colormap maximum dBZ (default: 40)')
    parser.add_argument('--cmap', type=str, default='HomeyerRainbow',
                        help='Background colormap. Py-ART reflectivity maps '
                             '(e.g. HomeyerRainbow, NWSRef) are registered when '
                             'Py-ART is installed; falls back to "Spectral_r" '
                             'if the name is unavailable.')
    parser.add_argument('--quiver_spacing_km', type=float, default=20.0,
                        help='Approximate quiver spacing in km (default: 20)')
    parser.add_argument('--quiver_scale', type=float, default=400.0,
                        help='matplotlib quiver "scale" — larger = shorter '
                             'arrows (default: 400)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second (default: 2)')
    parser.add_argument('--dpi', type=int, default=120,
                        help='Output DPI (default: 120)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8.0, 7.0],
                        help='Figure size W H in inches (default: 8 7)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GIF path')

    args = parser.parse_args()

    # Register Py-ART reflectivity colormaps if available, then validate the
    # requested name and gracefully fall back to a matplotlib default.
    try:
        import pyart  # noqa: F401  (registers its colormaps on import)
    except Exception:
        pass
    cmap = args.cmap
    if cmap not in plt.colormaps():
        print(f"Colormap '{cmap}' unavailable; using 'Spectral_r' instead.")
        cmap = 'Spectral_r'

    start_time = parse_optional_time(args.start_time)
    end_time = parse_optional_time(args.end_time)

    timestamps, files_a, files_b = find_time_pairs(
        args.grid_dir, args.index_a, args.index_b, start_time, end_time)
    if not timestamps:
        raise SystemExit(
            f"No complete grid pairs (suffix {args.index_a} & {args.index_b}) "
            f"in {args.grid_dir} for range {start_time}..{end_time}")
    print(f"Found {len(timestamps)} time step(s) with both grid files.")
    print(f"Level: {args.level}  Output: {args.output}")

    build_animation(
        timestamps=timestamps,
        files_a=files_a,
        files_b=files_b,
        level=args.level,
        refl_name=args.refl_var,
        u_name=args.u_var,
        v_name=args.v_var,
        w_name=args.w_var,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=cmap,
        quiver_spacing_km=args.quiver_spacing_km,
        quiver_scale=args.quiver_scale,
        contour_vars=args.contour,
        contour_levels=args.contour_levels,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        fps=args.fps,
        output=args.output,
    )
