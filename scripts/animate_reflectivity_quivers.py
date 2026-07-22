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

By default the frame is drawn on a geographic map (using the grid's 2-D lon/lat
coordinates) with ocean shading, coastline, country and state/province borders,
and state/province name labels. Pass ``--no-map`` for the original radar-relative
x/y-km plot, and ``--map_scale {10m,50m,110m}`` to pick the Natural Earth
feature resolution::

    python animate_reflectivity_quivers.py \
        --grid_dir /path/to/grids --map_scale 10m \
        --output reflectivity_quivers_map.gif
"""

import argparse
import glob
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Cartopy is only required for the geographic map overlay (--map). Import it
# lazily-tolerantly so the plain x/y-km path still runs where it is unavailable.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    _HAVE_CARTOPY = True
except Exception:
    ccrs = cfeature = shapereader = None
    _HAVE_CARTOPY = False


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

        # 2-D geographic coordinates (degrees) for the map overlay.
        lon2d = np.asarray(ds_a['lon'].values, dtype=float)
        lat2d = np.asarray(ds_a['lat'].values, dtype=float)

        refl_a = np.asarray(ds_a[refl_name].isel(time=0, z=level).values, dtype=float)
        refl_b = np.asarray(ds_b[refl_name].isel(time=0, z=level).values, dtype=float)
        # Element-wise max; a valid value on either side beats a NaN.
        refl = np.fmax(refl_a, refl_b)

        u = np.asarray(ds_a[u_name].isel(time=0, z=level).values, dtype=float)
        v = np.asarray(ds_a[v_name].isel(time=0, z=level).values, dtype=float)
        w = None
        if w_name in ds_a:
            w = np.asarray(ds_a[w_name].isel(time=0, z=level).values, dtype=float)
        return dict(x_km=x_km, y_km=y_km, lon2d=lon2d, lat2d=lat2d,
                    z_m=z_m, refl=refl, u=u, v=v, w=w)
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
    'speed': dict(colors='k', linewidth=2, label='wind speed [m/s]'),
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


def lowpass_field(field, sigma):
    """Gaussian low-pass filter a 2-D field, ignoring NaNs.

    ``sigma`` is the Gaussian standard deviation in grid cells; a value of 0
    (or None) returns the field unchanged. NaNs are handled with normalized
    convolution so missing points neither contribute to nor bleed into their
    neighbours, and points that were NaN stay NaN.
    """
    if not sigma or sigma <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(field, dtype=float)
    nan = ~np.isfinite(arr)
    weight = (~nan).astype(float)
    smoothed = gaussian_filter(np.where(nan, 0.0, arr), sigma=sigma,
                               mode='nearest')
    norm = gaussian_filter(weight, sigma=sigma, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = smoothed / norm
    out[nan] = np.nan
    return out


def draw_contours(ax, xc, yc, data, contour_vars, contour_levels,
                  transform=None, valid_mask=None, contour_smooth=0):
    """Overlay line contours of the requested wind components.

    ``xc``/``yc`` are the contour coordinate arrays (1-D x/y km, or 2-D lon/lat
    when ``transform`` is a cartopy CRS). ``valid_mask`` is an optional boolean
    array (True where valid); contours are restricted to those points so they
    are only drawn over regions with valid reflectivity echoes. When
    ``contour_smooth`` > 0 each field is Gaussian low-pass filtered (sigma in
    grid cells) before contouring. Returns a list of (label, Line2D-proxy)
    legend handles for the components actually drawn.
    """
    from matplotlib.lines import Line2D

    extra = {'transform': transform} if transform is not None else {}
    handles = []
    for name in contour_vars:
        field = contour_field(name, data)
        if field is None or not np.any(np.isfinite(field)):
            continue
        field = lowpass_field(field, contour_smooth)
        masked = np.ma.masked_invalid(field)
        if valid_mask is not None:
            masked = np.ma.masked_where(~valid_mask, masked)
            if not np.any(~masked.mask):
                continue
        style = CONTOUR_STYLE[name]
        cs = ax.contour(
            xc, yc, masked,
            levels=contour_levels, colors=style['colors'],
            linewidths=1.0, **extra)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
        handles.append(Line2D([0], [0], color=style['colors'],
                              lw=1.5, label=style['label']))
    return handles


# Fixed observation sites to mark on the map (station code, name, lat N, lon E).
# MVCO1 (South Beach) and MVCO2 (Shore Lab) are ~1 km apart, so they are shown
# as a single "MVCO" marker at their midpoint.
SITES = [
    ('NANT', 'Nantucket', 41.242, -70.125),
    ('MVCO', "Martha's Vineyard", 41.355, -70.526),
    ('BLOC', 'Block Island', 41.168, -71.580),
    ('RHOD', 'Narragansett', 41.445, -71.436),
    ('CACO', 'Nausett', 42.03, -70.049),
]


def draw_sites(ax, extent):
    """Plot the fixed observation sites (marker + station-code label).

    Only sites whose lon/lat fall inside ``extent`` (lon_min, lon_max, lat_min,
    lat_max) are drawn.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    for code, _name, lat, lon in SITES:
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        ax.plot(lon, lat, marker='*', markersize=11, color='magenta',
                markeredgecolor='black', markeredgewidth=0.6,
                transform=ccrs.PlateCarree(), zorder=6)
        txt = ax.text(lon, lat + 0.03, code, transform=ccrs.PlateCarree(),
                      fontsize=7, fontweight='bold', ha='center', va='bottom',
                      color='magenta', zorder=6)
        txt.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white'),
            path_effects.Normal()])


def add_map_features(ax, scale):
    """Add ocean/land shading, coastline, and country/state borders.

    Re-added every frame because ``ax.clear()`` wipes them; Natural Earth data
    is cached by cartopy after the first use so this stays cheap. ``scale`` is
    a Natural Earth resolution ('10m', '50m', or '110m').
    """
    ax.add_feature(cfeature.OCEAN.with_scale(scale), zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale(scale), zorder=0, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.8)
    ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.5,
                   edgecolor='gray')
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


def label_states(ax, extent, scale):
    """Label states/provinces whose center falls inside ``extent``.

    ``extent`` is (lon_min, lon_max, lat_min, lat_max) in degrees. Names come
    from the Natural Earth ``admin_1_states_provinces`` shapefile.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    shp = shapereader.natural_earth(
        resolution=scale, category='cultural',
        name='admin_1_states_provinces')
    for rec in shapereader.Reader(shp).records():
        name = rec.attributes.get('name') or rec.attributes.get('name_en')
        if not name:
            continue
        pt = rec.geometry.representative_point()
        if lon_min <= pt.x <= lon_max and lat_min <= pt.y <= lat_max:
            txt = ax.text(pt.x, pt.y, name, transform=ccrs.PlateCarree(),
                          fontsize=7, ha='center', va='center', color='black',
                          zorder=5)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='white'),
                path_effects.Normal()])


def draw_frame(fig, ax, cbar_ax, data, vmin, vmax, cmap,
               quiver_spacing_km, quiver_scale, title,
               contour_vars=(), contour_levels=None, contour_smooth=0,
               use_map=False, map_scale='50m', show_sites=True):
    """Render one frame in-place on the provided axes.

    When ``use_map`` is True, ``ax`` must be a cartopy GeoAxes and the frame is
    drawn in lon/lat (transform=PlateCarree) over ocean/land/border features.
    Otherwise the frame is drawn in radar-relative x/y km.
    """
    refl = data['refl']
    u = data['u']
    v = data['v']

    if use_map:
        lon2d = data['lon2d']
        lat2d = data['lat2d']
        # Stride from the mean grid spacing in km (x/y are regular).
        dx_km = data['x_km'][1] - data['x_km'][0]
        dy_km = data['y_km'][1] - data['y_km'][0]
        sx, sy = quiver_stride(quiver_spacing_km, dx_km, dy_km)
        X, Y = lon2d, lat2d
    else:
        x_km = data['x_km']
        y_km = data['y_km']
        dx_km = x_km[1] - x_km[0]
        dy_km = y_km[1] - y_km[0]
        sx, sy = quiver_stride(quiver_spacing_km, dx_km, dy_km)
        X, Y = np.meshgrid(x_km, y_km)

    # Mask quivers to points with a valid combined reflectivity value.
    valid = np.isfinite(refl)
    sub = (slice(None, None, sy), slice(None, None, sx))
    qmask = valid[sub]

    ax.clear()

    transform = None
    if use_map:
        transform = ccrs.PlateCarree()
        add_map_features(ax, map_scale)

    mesh_kw = {'transform': transform} if transform is not None else {}
    mesh = ax.pcolormesh(
        X if use_map else data['x_km'],
        Y if use_map else data['y_km'],
        np.ma.masked_invalid(refl),
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', **mesh_kw)
    ax.quiver(
        X[sub][qmask], Y[sub][qmask],
        u[sub][qmask], v[sub][qmask],
        scale=quiver_scale, color='black', width=0.0022, pivot='middle',
        **mesh_kw)

    contour_x = X if use_map else data['x_km']
    contour_y = Y if use_map else data['y_km']
    handles = draw_contours(ax, contour_x, contour_y, data, contour_vars,
                            contour_levels, transform=transform,
                            valid_mask=valid, contour_smooth=contour_smooth)
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8,
                  framealpha=0.85)

    if use_map:
        extent = [float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
                  float(np.nanmin(lat2d)), float(np.nanmax(lat2d))]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        label_states(ax, extent, map_scale)
        if show_sites:
            draw_sites(ax, extent)
    else:
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
                    figsize, dpi, fps, output, contour_smooth=0,
                    use_map=False, map_scale='50m', show_sites=True):
    subplot_kw = ({'projection': ccrs.PlateCarree()} if use_map else None)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
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
        title = (f'$Z_e$ + winds — '
                 f'z={data["z_m"]:.0f} m — '
                 f'{ts:%Y-%m-%d %H:%M UTC}')
        draw_frame(fig, ax, cbar_ax, data, vmin, vmax, cmap,
                   quiver_spacing_km, quiver_scale, title,
                   contour_vars=contour_vars, contour_levels=contour_levels,
                   contour_smooth=contour_smooth,
                   use_map=use_map, map_scale=map_scale, show_sites=show_sites)
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
    parser.add_argument('--contour_smooth', type=float, default=0.0,
                        metavar='SIGMA',
                        help='Apply a Gaussian low-pass filter to the contour '
                             'fields before contouring; SIGMA is the standard '
                             'deviation in grid cells (e.g. 1.5). NaNs are '
                             'ignored. 0 disables smoothing (default: 0)')
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
    parser.add_argument('--map', dest='map', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Overlay data on a geographic map with coastline, '
                             'country/state borders, ocean shading and '
                             'state/province labels (default: on; use --no-map '
                             'for the plain x/y-km plot)')
    parser.add_argument('--map_scale', type=str, default='50m',
                        choices=['10m', '50m', '110m'],
                        help='Natural Earth feature resolution for --map '
                             '(default: 50m)')
    parser.add_argument('--sites', dest='sites',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Mark the fixed observation sites (NANT, MVCO1, '
                             'MVCO2, BLOC, RHOD, CACO) on the map '
                             '(default: on; --map only)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GIF path')

    args = parser.parse_args()

    if args.map and not _HAVE_CARTOPY:
        raise SystemExit(
            "The --map overlay requires cartopy, which is not importable. "
            "Install it (e.g. `conda install -c conda-forge cartopy`) or rerun "
            "with --no-map for the plain x/y-km plot.")

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
        contour_smooth=args.contour_smooth,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        fps=args.fps,
        output=args.output,
        use_map=args.map,
        map_scale=args.map_scale,
        show_sites=args.sites,
    )
