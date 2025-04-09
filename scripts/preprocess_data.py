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
from dask_jobqueue import PBSCluster
from scipy.spatial import KDTree

radar = sys.argv[1]
date = sys.argv[2]
rad_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/', radar)
out_proc_path = os.path.join(
    '/lcrc/group/earthscience/rjackson/Earnest/pre_processed/', radar)

if not os.path.exists(out_proc_path):
    os.makedirs(out_proc_path)


def make_quicklooks(file):
    base, name = os.path.split(file)

    if os.path.exists(os.path.join(out_proc_path, name)):
        return
    try:
        rad = pyart.io.read(file)
    except TypeError:
        return
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
                rad.fields['velocity']['data'][
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
            if field in rad.fields.keys():
                offset = min([dest_slice.start, dest_slice.stop])
                rad_dest.fields[field]['data'][indices + offset, :] = rad.fields[field]['data'][int(sl.start):int(sl.stop), :]

    pyart.io.write_cfradial(os.path.join(out_proc_path, name + '.nc'), rad_dest)
    del fig
    print("Preprocessing for %s completed!" % file)

if __name__ == "__main__":
    file_list = sorted(glob.glob(rad_path + f"/{radar}{date}*"))
    print(rad_path + f"/{radar}{date}*")
    print(file_list)
    cluster = PBSCluster(processes=6, cores=36, memory='128GB', walltime='6:00:00',
            account="rainfall") 
    cluster.scale(24)
    with Client(cluster) as c:
        c.wait_for_workers(6)
        results = c.map(make_quicklooks, file_list)
        wait(results)
