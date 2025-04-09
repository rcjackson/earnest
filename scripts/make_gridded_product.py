import pyart
import matplotlib.pyplot as plt
import cmweather
import sys
import xarray as xr
import os

from distributed import Client, wait, LocalCluster
from dask_jobqueue import PBSCluster
from glob import glob

radar = sys.argv[1]
out_plot_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/quicklooks_grid/', radar)
out_proc_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/grids/', radar)
in_rad_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/processed/', radar)

grid_z_range = (0., 15000)
grid_x_range = (-125000., 125000.)
grid_y_range = (-125000., 125000.)
grid_resolution_h = 1000.
grid_resolution_v = 500.

def calculate_grid_points(grid_limits, resolution):
    nx = int((grid_limits[1] - grid_limits[0]) / resolution) + 1
    return nx

grid_spec = (calculate_grid_points(grid_z_range, grid_resolution_v),
             calculate_grid_points(grid_y_range, grid_resolution_h),
             calculate_grid_points(grid_x_range, grid_resolution_h))

if not os.path.exists(out_plot_path):
    os.makedirs(out_plot_path)

if not os.path.exists(out_proc_path):
    os.makedirs(out_proc_path)

def grid_and_quicklooks(file_name):
    base, name = os.path.split(file_name)
    out_file_path = os.path.join(out_proc_path, name)
    if os.path.exists(out_file_path):
        return
    radar = pyart.io.read(file_name)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid('corrected_velocity_region_based')
    grid = pyart.map.grid_from_radars([radar], grid_spec, (grid_z_range, grid_y_range, grid_x_range),
        fields = ["reflectivity", "corrected_velocity_region_based", "corrected_velocity_unravel"],
        gatefilters=[gatefilter],
        min_radius=250., weighting_function='Cressman')

    fig, ax = plt.subplots(3, 3, figsize=(15, 9))
    xr_grid = grid.to_xarray()
    xr_grid['corrected_velocity_unravel'].sel(z=1000., method='nearest').plot(ax=ax[0, 0],
                                              vmin=-60, vmax=60, cmap='balance')
    xr_grid['corrected_velocity_unravel'].sel(y=0., method='nearest').plot(ax=ax[1, 0],
                                              vmin=-60, vmax=60, cmap='balance')
    xr_grid['corrected_velocity_unravel'].sel(x=0., method='nearest').plot(ax=ax[2, 0],
                                              vmin=-60, vmax=60, cmap='balance')
    xr_grid['corrected_velocity_region_based'].sel(z=1000., method='nearest').plot(ax=ax[0, 1],
                                                   vmin=-60, vmax=60, cmap='balance')
    xr_grid['corrected_velocity_region_based'].sel(y=0., method='nearest').plot(ax=ax[1, 1],
                                                   vmin=-60, vmax=60, cmap='balance')
    xr_grid['corrected_velocity_region_based'].sel(x=0., method='nearest').plot(ax=ax[2, 1],
                                                   vmin=-60, vmax=60, cmap='balance')
    xr_grid['reflectivity'].sel(z=1000., method='nearest').plot(ax=ax[0, 2],
                                vmin=-20, vmax=80, cmap='ChaseSpectral')
    xr_grid['reflectivity'].sel(y=0., method='nearest').plot(ax=ax[1, 2],
                                vmin=-20, vmax=80, cmap='ChaseSpectral')
    xr_grid['reflectivity'].sel(x=0., method='nearest').plot(ax=ax[2, 2],
                                vmin=-20, vmax=80, cmap='ChaseSpectral')
    fig.tight_layout()
    out_png_path = os.path.join(out_plot_path, name + '.png')
    fig.savefig(out_png_path, bbox_inches='tight')
    out_file_path = os.path.join(out_proc_path, name)
    pyart.io.write_grid(out_file_path, grid)
    plt.close(fig)
  
if __name__ == "__main__":    
    file_list = sorted(glob(in_rad_path + '/*.nc'))
    # cluster = PBSCluster(processes=6, cores=36, memory='128GB', walltime='6:00:00',
    #        account='rainfall')
    #cluster.scale(24)
    cluster = LocalCluster(n_workers=6)
    with Client(cluster) as c:
        c.wait_for_workers(6)
        results = c.map(grid_and_quicklooks, file_list)
        wait(results)

