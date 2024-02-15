import pyart
import cmweather
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cartopy.crs as ccrs

from distributed import Client, wait
from dask_jobqueue import SLURMCluster
from scipy.spatial import KDTree

radar = sys.argv[1]
out_plot_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/quicklooks_dealias/', radar)
rad_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/', radar)
out_proc_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/processed/', radar)

if not os.path.exists(out_proc_path):
    os.makedirs(out_proc_path)

if not os.path.exists(out_plot_path):
    os.makedirs(out_plot_path)

def make_quicklooks(file):    
    try:
        rad = pyart.io.read(file)
    except TypeError:
        return
    base, name = os.path.split(file)
    # Let's actually have a coherent NEXRAD scan
    which_sweeps_vel = []
    which_elevations = []
    for i, s in enumerate(rad.iter_slice()):
        sum_vel = np.ma.sum(rad.fields['velocity']['data'][s, :])
        elevation = rad.fixed_angle['data'][i]
        if not np.ma.is_masked(sum_vel) and not elevation in which_elevations:
            which_sweeps_vel.append(i)
            which_elevations.append(elevation)
        
    fields_needed = ['differential_phase',
            'cross_correlation_ratio',
            'reflectivity',
            'differential_reflectivity']

    rad_dest = rad.extract_sweeps(which_sweeps_vel)
    sum_ref = np.ma.masked
    for i, sweep in enumerate(which_sweeps_vel):
        for cand_sweep in np.argwhere(
            np.isclose(rad.fixed_angle['data'], rad.fixed_angle['data'][sweep])):
            sl = rad.get_slice(cand_sweep)
            sum_ref = np.ma.sum(
                rad.fields['cross_correlation_ratio']['data'][
                    int(sl.start):int(sl.stop), :])
            src_slice = rad.get_slice(sweep)
            if not np.ma.is_masked(sum_ref) and (sl.stop - sl.start) == (src_slice.stop - src_slice.start):
                break
        dest_slice = rad_dest.get_slice(i)
        dest_azi = rad_dest.azimuth['data'][dest_slice.start:dest_slice.stop]
        src_azi = rad.azimuth['data'][int(sl.start):int(sl.stop)]
        tree = KDTree(dest_azi[:, np.newaxis])
        neighbors, indices = tree.query(
            src_azi[:, np.newaxis], distance_upper_bound=1)
        for field in fields_needed:
            offset = min([dest_slice.start, dest_slice.stop])
            rad_dest.fields[field]['data'][indices + offset, :] = rad.fields[field]['data'][int(sl.start):int(sl.stop), :]
    rad_gatefilter = pyart.filters.GateFilter(rad_dest)
    del rad
    text_phase = pyart.bridge.texture_of_complex_phase(rad_dest)
    rad_dest.add_field('texture_of_differential_phase',
        text_phase, replace_existing=True)
    disp = pyart.graph.RadarMapDisplay(rad_dest)
    gatefilter = pyart.filters.GateFilter(rad_dest)
    gatefilter.exclude_above('texture_of_differential_phase', 50)
    gatefilter.exclude_below('reflectivity', 0)
    gatefilter = pyart.correct.despeckle_field(
        rad_dest, 'velocity', gatefilter=gatefilter)
    corrected_velocity = pyart.correct.dealias_region_based(
        rad_dest, gatefilter=gatefilter, centered=True)
    rad_dest.add_field('corrected_velocity', corrected_velocity, replace_existing=True)
    num_rows = int(np.ceil(rad_dest.nsweeps/2))
    fig, ax = plt.subplots(num_rows, 2, figsize=(8, num_rows*3))
    for i in range(rad_dest.nsweeps):
        disp.plot_ppi('corrected_velocity', i, ax=ax[int(i / 2), i % 2],
                 vmin=-70, vmax=70,
                 cmap='pyart_balance', colorbar_label='Doppler velocity [m/s]')
        ax[int(i / 2), i % 2].set_xlim([-100, 100])
        ax[int(i / 2), i % 2].set_ylim([-100, 100])
        ax[int(i / 2), i % 2].set_xlabel('X [km]')
        ax[int(i / 2), i % 2].set_ylabel('Y [km]')
        title_split = ax[int(i / 2), i % 2].get_title().split("\n")
        ax[int(i / 2), i % 2].set_title(title_split[0])
    fig.tight_layout()

    fig.savefig(os.path.join(out_plot_path, name + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    pyart.io.write_cfradial(os.path.join(out_proc_path, name + '.nc'), rad_dest)
    del fig
    print("Quicklooks/Dealiasing for %s completed!" % file)

if __name__ == "__main__":
    file_list = sorted(glob.glob(rad_path + '/*'))
    cluster = SLURMCluster(processes=6, cores=36, memory='128GB', walltime='6:00:00') 
    cluster.scale(24)
    with Client(cluster) as c:
        c.wait_for_workers(6)
        results = c.map(make_quicklooks, file_list)
        wait(results)
