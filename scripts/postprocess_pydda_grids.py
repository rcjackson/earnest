"""
postprocess_pydda_grids.py
---------------------------
Post-process the PyDDA retrieval grids written by
``do_pydda_retrieval_sounding_back.py``.

For each grid:

* Gaussian low-pass filter (sigma=1.5 grid cells by default, same filter as
  ``lowpass_field`` in ``animate_reflectivity_quivers.py``) is applied to the
  ``u``, ``v``, and ``w`` fields, independently at each time/height level.
  NaNs are ignored (normalized convolution) so missing points neither
  contribute to nor bleed into their neighbours.
* Wind speed (``spd``) and direction (``dir``, meteorological convention —
  degrees the wind is blowing *from*, 0=N/90=E/180=S/270=W) are (re)computed
  from the filtered ``u``/``v`` so they stay consistent with the smoothed
  wind field.
* Only ``reflectivity``, ``u``, ``v``, ``w``, ``spd``, and ``dir`` are kept;
  every other variable (HRRR fields, radar moments, point geometry, etc.) is
  dropped.

Example
-------
    python postprocess_pydda_grids.py \
        --grid_dir /projects/storm/rjackson/wfip3/multidoppler_grids/grids \
        --output_dir /projects/storm/rjackson/wfip3/multidoppler_grids/grids_post \
        --sigma 1.5
"""

import argparse
import glob
import os
import re

import numpy as np
import xarray as xr

GRID_INDEX_RE = re.compile(r'^(.*)_(\d+)\.nc$')

KEEP_VARS = ('reflectivity', 'u', 'v', 'w', 'spd', 'dir')


def lowpass_field(field, sigma):
    """Gaussian low-pass filter a field over its last two axes, ignoring NaNs.

    ``field`` may be 2-D (y, x) or N-D with (y, x) as the trailing two axes
    (e.g. (time, z, y, x)); only those trailing axes are smoothed. ``sigma``
    is the Gaussian standard deviation in grid cells; a value of 0 (or None)
    returns the field unchanged. NaNs are handled with normalized
    convolution so missing points neither contribute to nor bleed into their
    neighbours, and points that were NaN stay NaN.
    """
    if not sigma or sigma <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(field, dtype=float)
    sigma_nd = [0] * (arr.ndim - 2) + [sigma, sigma]
    nan = ~np.isfinite(arr)
    weight = (~nan).astype(float)
    smoothed = gaussian_filter(np.where(nan, 0.0, arr), sigma=sigma_nd,
                               mode='nearest')
    norm = gaussian_filter(weight, sigma=sigma_nd, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = smoothed / norm
    out[nan] = np.nan
    return out


def postprocess_grid(ds, sigma=1.5):
    """Return a post-processed copy of a PyDDA retrieval grid dataset.

    Smooths ``u``/``v``/``w`` with ``lowpass_field`` (sigma in grid cells),
    recomputes ``spd``/``dir`` from the smoothed ``u``/``v``, and drops every
    variable other than ``reflectivity``, ``u``, ``v``, ``w``, ``spd``, and
    ``dir``.
    """
    ds = ds.copy()

    for var in ('u', 'v', 'w'):
        if var in ds:
            attrs = ds[var].attrs
            ds[var] = (ds[var].dims, lowpass_field(ds[var].values, sigma), attrs)

    u, v = ds['u'], ds['v']
    ds['spd'] = np.hypot(u, v)
    ds['spd'].attrs = {'long_name': 'Wind speed', 'units': 'm s-1'}

    ds['dir'] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    ds['dir'].attrs = {'long_name': 'Wind direction (from)', 'units': 'degrees'}

    drop_vars = [v for v in ds.data_vars if v not in KEEP_VARS]
    return ds.drop_vars(drop_vars)


def merge_max_reflectivity(processed_grids):
    """Merge post-processed grids that share one wind field but have their
    own ``reflectivity`` (e.g. the ``_0``/``_1`` per-radar grids written by
    ``do_pydda_retrieval_sounding_back.py``).

    ``u``/``v``/``w``/``spd``/``dir`` are taken from the first grid (they are
    the same retrieved wind field broadcast to every grid); ``reflectivity``
    is the elementwise maximum across all of them, ignoring NaNs.
    """
    merged = processed_grids[0].copy()
    attrs = processed_grids[0]['reflectivity'].attrs
    refl_stack = xr.concat([g['reflectivity'] for g in processed_grids],
                           dim='_grid')
    merged['reflectivity'] = refl_stack.max(dim='_grid', skipna=True)
    merged['reflectivity'].attrs = attrs
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Post-process PyDDA retrieval grids: Gaussian-smooth '
                    'u/v/w, recompute spd/dir, and keep only reflectivity, '
                    'u, v, w, spd, dir.')
    parser.add_argument('--grid_dir', type=str, required=True,
                        help='Directory of grid_YYYYMMDD_HHMMSS_*.nc files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: overwrite in place)')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Gaussian filter standard deviation in grid '
                             'cells (default: 1.5)')
    parser.add_argument('--pattern', type=str, default='grid_*.nc',
                        help='Glob pattern for grid files (default: grid_*.nc)')
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.grid_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.grid_dir}")

    # Group files sharing a common base name (e.g. grid_20240213_065753_0.nc
    # and grid_20240213_065753_1.nc share the base grid_20240213_065753) so
    # that _0/_1 pairs can be merged into a single max-reflectivity grid.
    groups = {}
    order = []
    for f in files:
        base = os.path.basename(f)
        match = GRID_INDEX_RE.match(base)
        key, idx = (match.group(1), int(match.group(2))) if match else (base, None)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][idx] = f

    for key in order:
        idx_map = groups[key]
        if set(idx_map) == {0, 1}:
            processed_grids = []
            for idx in (0, 1):
                with xr.open_dataset(idx_map[idx]) as ds:
                    processed_grids.append(postprocess_grid(ds, sigma=args.sigma).load())
            merged = merge_max_reflectivity(processed_grids)
            out_dir = args.output_dir or args.grid_dir
            out_path = os.path.join(out_dir, f"{key}.nc")
            merged.to_netcdf(out_path)
            print(f"Post-processed (merged max reflectivity): {out_path}")
            continue

        for f in idx_map.values():
            out_path = (os.path.join(args.output_dir, os.path.basename(f))
                       if args.output_dir else f)
            with xr.open_dataset(f) as ds:
                processed = postprocess_grid(ds, sigma=args.sigma)
                processed.load()
            if out_path == f:
                tmp_path = out_path + '.tmp'
                processed.to_netcdf(tmp_path)
                os.replace(tmp_path, out_path)
            else:
                processed.to_netcdf(out_path)
            print(f"Post-processed: {out_path}")
