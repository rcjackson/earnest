import pyart
import cmweather
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cartopy.crs as ccrs
import unravel

from distributed import Client, wait
from dask_jobqueue import SLURMCluster
from scipy.spatial import KDTree

radar = sys.argv[1]
out_plot_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/quicklooks_dealias/', radar)
rad_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/pre_processed/', radar)
out_proc_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/processed/', radar)

if not os.path.exists(out_proc_path):
    os.makedirs(out_proc_path)

if not os.path.exists(out_plot_path):
    os.makedirs(out_plot_path)

def make_quicklooks(file, prev_file):    
    try:
        rad = pyart.io.read(file)
    except TypeError:
        return
    base, name = os.path.split(file)
    disp = pyart.graph.RadarMapDisplay(rad_dest)
    gatefilter = pyart.filters.GateFilter(rad_dest)
    gatefilter.exclude_above('texture_of_differential_phase', 50)
    gatefilter.exclude_below('reflectivity', 0)
    gatefilter = pyart.correct.despeckle_field(
        rad_dest, 'velocity', gatefilter=gatefilter)
    corrected_velocity = pyart.correct.dealias_region_based(
        rad_dest, gatefilter=gatefilter, centered=True, skip_between_rays=100, skip_along_rays=1000)
    if prev_file is None:
        use_previous = False
    else:
        rad_prev = pyart.io.read(prev_file)
        if prev_file.fields['velocity'].shape == file.fields['velocity'].shape:
            rad_dest.fields['previous_velocity'] = file.fields['corrected_velocity_region_based']
            use_previous = True
        else:
            use_previous= False
    if use_previous is False:
        gatefilter = pyart.correct.despeckle_field(
            rad_dest, 'velocity', gatefilter=gatefilter)
    else:
        gatefilter = pyart.correct.despeckle_field(
            rad_dest, 'velocity', gatefilter=gatefilter, ref_vel_field='previous_velocity')
    rad_dest.add_field('corrected_velocity_region_based', corrected_velocity, replace_existing=True)
    
    dealiased_velocity = unravel.unravel_3D_pyart(
        rad_dest, velname='velocity', dbzname='reflectivity', do_3d=True)
    rad_dest.add_field_like('velocity', 'corrected_velocity_unravel',
        dealiased_velocity, replace_existing=True)
    num_rows = int(np.ceil(rad_dest.nsweeps/2))
    fig, ax = plt.subplots(num_rows, 4, figsize=(16, num_rows*3))
    for i in range(rad_dest.nsweeps):
        disp.plot_ppi('corrected_velocity_region_based', i, ax=ax[int(i / 2), i % 2],
                 vmin=-70, vmax=70,
                 cmap='pyart_balance', colorbar_label='Region Doppler velocity [m/s]')
        ax[int(i / 2), i % 2].set_xlim([-150, 150])
        ax[int(i / 2), i % 2].set_ylim([-150, 150])
        ax[int(i / 2), i % 2].set_xlabel('X [km]')
        ax[int(i / 2), i % 2].set_ylabel('Y [km]')
        title_split = ax[int(i / 2), i % 2].get_title().split("\n")
        ax[int(i / 2), i % 2].set_title(title_split[0])
        disp.plot_ppi('corrected_velocity_unravel', i, ax=ax[int(i / 2), (i % 2) + 2],
                 vmin=-70, vmax=70,
                 cmap='pyart_balance', colorbar_label='UNRAVEL Doppler velocity [m/s]')
        ax[int(i / 2), (i % 2) + 2].set_xlim([-150, 150])
        ax[int(i / 2), (i % 2) + 2].set_ylim([-150, 150])
        ax[int(i / 2), (i % 2) + 2].set_xlabel('X [km]')
        ax[int(i / 2), (i % 2) + 2].set_ylabel('Y [km]')
        title_split = ax[int(i / 2), (i % 2) + 2].get_title().split("\n")
        ax[int(i / 2), (i % 2) + 2].set_title(title_split[0])


    fig.tight_layout()

    fig.savefig(os.path.join(out_plot_path, name + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    pyart.io.write_cfradial(os.path.join(out_proc_path, name + '.nc'), rad_dest)
    del fig
    print("Quicklooks/Dealiasing for %s completed!" % file)

if __name__ == "__main__":
    file_list = sorted(glob.glob(rad_path + '/*'))
    for i in range(len(file_list)):
        if i > 1:
            make_quicklooks(file_list[i], file_list[i - 1])
        else:
            make_quicklooks(file_list[i], None)
