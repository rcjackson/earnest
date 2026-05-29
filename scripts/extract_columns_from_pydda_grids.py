import argparse
import xarray as xr
import sys
import os
import numpy as np
    
sites = ["bloc", "nant", "rhod", "caco"]

def extract_column(ds: xr.Dataset, target_lat: float, target_lon: float, site_name: str, variables: list):
      """
      Extract the column (all z, all time) nearest to (target_lat,
  target_lon).

      Returns a dataset with dims (time, z) for all variables that have
  (time, z, y, x).
      """
      lat2d = ds["lat"].values
      lon2d = ds["lon"].values

      # Equirectangular approximation is plenty for picking a single cell.
      # Scale longitude by cos(lat) so degrees are roughly equal-distance.
      coslat = np.cos(np.deg2rad(target_lat))
      dlat = lat2d - target_lat
      dlon = (lon2d - target_lon) * coslat
      dist2 = dlat**2 + dlon**2

      iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

      column = ds.isel(y=iy, x=ix)

      # Attach the actual lat/lon of the chosen cell for traceability.

      # Attach the actual lat/lon of the chosen cell for traceability.
      column = column.assign_attrs(
          extracted_lat=float(lat2d[iy, ix]),
          extracted_lon=float(lon2d[iy, ix]),
          requested_lat=float(target_lat),
          requested_lon=float(target_lon),
          y_index=int(iy),
          x_index=int(ix),
      )
      for var in column.data_vars:
          if var not in variables:
              column = column.drop(var)
          else:
              column[f"{var}_{site_name}"] = column[var]
              column = column.drop(var)
      column = column.drop(["lat", "lon"])
      return column

def main():
    args = parse_args()
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    grid_path = os.path.join(args.root_data_dir, f"grids_nearest_100iter_co1000_fxx{args.forecast_hour}")
    print(f"Grid path: {grid_path}")
    multidoppler_ds = xr.open_mfdataset(os.path.join(grid_path, f"*{args.date}*_0.nc"))
    columns = []
    for s in sites:
        windprof_column_name = os.path.join('/projects/storm/rjackson/wfip3/windprof',
        f"{s}.windprof.z01.c1.{args.date}.000000.nc")
        windprof_column = xr.open_dataset(windprof_column_name)
        site_lat = windprof_column.attrs["latitude"]
        site_lon = windprof_column.attrs["longitude"]
        windprof_column.close()
        print(f"Processing column over {s}")
        site_column = extract_column(multidoppler_ds, site_lat, site_lon, s, 
                ["u", "v", "w", "U_hrrr", "V_hrrr", "V_hrrr", "spd", "dir", "reflectivity",
                    "corrected_velocity_unravel"])
        columns.append(site_column)
    out_ds = xr.merge(columns, compat="override")
    out_ds.to_netcdf(os.path.join(args.out_data_dir, 
        f"multidoppler_wind_columns_nearest_100iter_f{args.forecast_hour}_{args.date}.nc"))
    multidoppler_ds.close()
    
    return

def parse_args():
      argparser = argparse.ArgumentParser(
          description="Parse a forecast hour from the command line."
      )

      argparser.add_argument(
          "-f", "--forecast-hour",
          type=int,
          required=True,
          help="Forecast hour (integer, e.g. 0, 6, 12, 24).",
      )
      argparser.add_argument(
              "-d", "--date",
              type=str,
              required=True,
              help="Date in YYYYMMDD")
      argparser.add_argument(
              "--root_data_dir",
              type=str,
              required=False,
              default="/projects/storm/rjackson/wfip3/multidoppler_grids")
      argparser.add_argument(
              "--out_data_dir",
              type=str,
              required=False,
              default="/projects/storm/rjackson/wfip3/multidoppler_columns")

      return argparser.parse_args()

if __name__ == "__main__":
   main()
   


