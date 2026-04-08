"""
nexrad_animation.py
-------------------
Animate NEXRAD PPI sweeps for KOKX (Upton, NY) and KBOX (Taunton, MA)
radar sites using PyART and cartopy.

Usage
-----
    from nexrad_animation import animate_nexrad_kokx_kbox
    import glob, sorted

    kokx_files = sorted(glob.glob("/data/KOKX*.ar2v"))
    kbox_files  = sorted(glob.glob("/data/KBOX*.ar2v"))

    anim = animate_nexrad_kokx_kbox(
        kokx_files,
        kbox_files,
        output_path="kokx_kbox.gif",
    )
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pyart
from cartopy.feature import NaturalEarthFeature


# Default map extent covering both KOKX (Upton, NY) and KBOX (Taunton, MA).
# KOKX: 40.866 N, 72.864 W  |  KBOX: 41.956 N, 71.137 W
_DEFAULT_EXTENT = [-75.5, -68.5, 39.0, 44.5]  # [lon_min, lon_max, lat_min, lat_max]


def _parse_radar_time(radar):
    """Return a :class:`datetime` for the first ray of a PyART Radar object."""
    units = radar.time["units"]  # e.g. "seconds since 2023-05-10T18:00:00Z"
    ref_str = units.split("since ")[-1].strip().rstrip("Z")
    try:
        ref_dt = datetime.strptime(ref_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        ref_dt = datetime.strptime(ref_str, "%Y-%m-%d %H:%M:%S")
    return ref_dt + timedelta(seconds=float(radar.time["data"][0]))


def _add_map_features(ax):
    """Overlay 10-m resolution geographic boundaries on a cartopy GeoAxes."""
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=6
    )
    ax.add_feature(
        cfeature.STATES.with_scale("10m"), linewidth=0.5, zorder=6
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"), linewidth=0.9, zorder=6
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="lightcyan",
        edgecolor="steelblue",
        linewidth=0.4,
        alpha=0.6,
        zorder=5,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"),
        edgecolor="steelblue",
        linewidth=0.3,
        alpha=0.5,
        zorder=5,
    )
    counties = NaturalEarthFeature(
        category="cultural",
        name="admin_2_counties",
        scale="10m",
        facecolor="none",
        edgecolor="gray",
    )
    ax.add_feature(counties, linewidth=0.25, zorder=5)


def _load_radar(filepath):
    """Read one NEXRAD archive file; returns None if *filepath* is None."""
    if filepath is None:
        return None
    return pyart.io.read_nexrad_archive(filepath)


def _draw_frame(fig, kokx_radar, kbox_radar, field, sweep, vmin, vmax, cmap, extent):
    """Clear *fig* and draw one animation frame with both radar panels."""
    lon_min, lon_max, lat_min, lat_max = extent
    proj = ccrs.PlateCarree()

    fig.clf()
    ax_kokx = fig.add_subplot(1, 2, 1, projection=proj)
    ax_kbox = fig.add_subplot(1, 2, 2, projection=proj)

    time_str = ""

    for ax, radar, label in (
        (ax_kokx, kokx_radar, "KOKX"),
        (ax_kbox, kbox_radar, "KBOX"),
    ):
        if radar is None:
            ax.set_extent(extent, crs=proj)
            _add_map_features(ax)
            ax.set_title(f"{label} — no data", fontsize=10)
            continue

        display = pyart.graph.RadarMapDisplay(radar)
        display.plot_ppi_map(
            field,
            sweep=sweep,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar_flag=True,
            title_flag=False,
            min_lon=lon_min,
            max_lon=lon_max,
            min_lat=lat_min,
            max_lat=lat_max,
            # Use PyART's built-in 10m features; we add extras below.
            resolution="10m",
        )

        t = _parse_radar_time(radar)
        if not time_str:
            time_str = t.strftime("%Y-%m-%d %H:%M UTC")

        elev = float(radar.fixed_angle["data"][sweep])
        ax.set_title(
            f"{label}  \u2014  {t.strftime('%H:%M UTC')}  ({elev:.1f}\u00b0 elev)",
            fontsize=10,
        )

        # Extra features PyART does not add by default (lakes, rivers, counties).
        ax.add_feature(
            cfeature.LAKES.with_scale("10m"),
            facecolor="lightcyan",
            edgecolor="steelblue",
            linewidth=0.4,
            alpha=0.6,
            zorder=5,
        )
        ax.add_feature(
            cfeature.RIVERS.with_scale("10m"),
            edgecolor="steelblue",
            linewidth=0.3,
            alpha=0.5,
            zorder=5,
        )
        counties = NaturalEarthFeature(
            category="cultural",
            name="admin_2_counties",
            scale="10m",
            facecolor="none",
            edgecolor="gray",
        )
        ax.add_feature(counties, linewidth=0.25, zorder=5)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

        del display  # radar is owned by the caller; free only the display

    fig.suptitle(f"NEXRAD — {time_str}", fontsize=13, y=1.01)
    fig.tight_layout()


def animate_nexrad_kokx_kbox(
    kokx_files,
    kbox_files,
    field="reflectivity",
    sweep=0,
    vmin=-20,
    vmax=70,
    cmap="pyart_NWSRef",
    figsize=(18, 8),
    output_path=None,
    fps=2,
    dpi=100,
    extent=None,
    max_workers=4,
    prefetch=None,
):
    """
    Produce a side-by-side animation of KOKX and KBOX NEXRAD PPI sweeps.

    Parameters
    ----------
    kokx_files : list of str
        Paths to KOKX NEXRAD Level-II archive files in chronological order.
    kbox_files : list of str
        Paths to KBOX NEXRAD Level-II archive files in chronological order.
    field : str, optional
        Radar field to display. Default ``'reflectivity'``.
    sweep : int, optional
        Sweep (tilt) index to display. Default ``0`` (lowest elevation).
    vmin : float, optional
        Colormap minimum value. Default ``-20`` dBZ.
    vmax : float, optional
        Colormap maximum value. Default ``70`` dBZ.
    cmap : str, optional
        Matplotlib or PyART colormap name. Default ``'pyart_NWSRef'``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches. Default ``(18, 8)``.
    output_path : str or None, optional
        Save destination for the animation. Use a ``.gif`` extension for a
        Pillow-written GIF or ``.mp4`` for an FFmpeg-written video. When
        *None* the animation object is returned without saving.
    fps : int, optional
        Frames per second. Default ``2``.
    dpi : int, optional
        Output resolution in dots per inch. Default ``100``.
    extent : list or None, optional
        Map extent ``[lon_min, lon_max, lat_min, lat_max]``. Defaults to a
        region that encompasses both KOKX and KBOX coverage areas.
    max_workers : int, optional
        Number of threads used for parallel file loading. Default ``4``.
    prefetch : int or None, optional
        Number of frames to load ahead of the current frame. Defaults to
        ``max_workers * 2``, which keeps all threads busy without holding
        too many radar objects in memory simultaneously.

    Returns
    -------
    anim : `matplotlib.animation.FuncAnimation`
        The animation object. Call ``plt.show()`` to display it interactively
        or use *output_path* to save it.

    Examples
    --------
    >>> import glob
    >>> kokx = sorted(glob.glob("/data/KOKX20230501_*.ar2v"))
    >>> kbox  = sorted(glob.glob("/data/KBOX20230501_*.ar2v"))
    >>> anim = animate_nexrad_kokx_kbox(kokx, kbox, output_path="loop.gif")
    """
    if extent is None:
        extent = _DEFAULT_EXTENT

    if prefetch is None:
        prefetch = max_workers * 2

    n_kokx = len(kokx_files)
    n_kbox = len(kbox_files)
    n_frames = max(n_kokx, n_kbox)

    if n_frames == 0:
        raise ValueError("Both kokx_files and kbox_files are empty.")

    # Bounded streaming pool: at most `prefetch` frames are in-flight at once.
    # Each entry in `pending` is (kokx_future, kbox_future); popping it after
    # rendering lets both radar objects be garbage-collected immediately.
    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending: dict = {}  # frame index -> (kokx_future, kbox_future)

    def _submit_if_needed(frame):
        if 0 <= frame < n_frames and frame not in pending:
            kokx_fp = kokx_files[frame] if frame < n_kokx else None
            kbox_fp = kbox_files[frame] if frame < n_kbox else None
            pending[frame] = (
                executor.submit(_load_radar, kokx_fp),
                executor.submit(_load_radar, kbox_fp),
            )

    # Prime the initial prefetch window.
    for f in range(min(prefetch, n_frames)):
        _submit_if_needed(f)

    fig = plt.figure(figsize=figsize)

    def update(frame):
        # Ensure current frame is submitted (handles loop restart at frame 0).
        _submit_if_needed(frame)
        # Slide the prefetch window forward.
        for f in range(frame + 1, min(frame + prefetch + 1, n_frames)):
            _submit_if_needed(f)
        # Block until this frame's files are ready, then discard the futures.
        kokx_f, kbox_f = pending.pop(frame)
        kokx_radar = kokx_f.result()
        kbox_radar = kbox_f.result()
        _draw_frame(
            fig, kokx_radar, kbox_radar, field, sweep, vmin, vmax, cmap, extent
        )
        # kokx_radar and kbox_radar go out of scope here and are freed.

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        repeat=True,
        blit=False,
    )
    # Keep the executor alive for the animation's lifetime.
    anim._executor = executor

    if output_path is not None:
        if output_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer, dpi=dpi)
        executor.shutdown(wait=False)
        print(f"Saved animation → {output_path}")

    return anim
