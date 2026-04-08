import pyart
import sys
import numpy as np
import glob
import xarray as xr
import os

from distributed import Client, LocalCluster
from datetime import datetime

def extract_vcp(fil):
    radar = pyart.io.read(fil)
    if "vcp_pattern" in radar.metadata.keys():
        vcp = radar.metadata["vcp_pattern"]
        time = np.datetime64(radar.time["units"].split(" ")[-1][:-1])

    del radar
    return time, vcp

if __name__ == "__main__":
    radar = sys.argv[1]
    rad_path = os.path.join(
        '/projects/storm/rjackson/wfip3/cfradial/', radar)
    rad_files = sorted(glob.glob(rad_path + '/*.nc'))
    with Client(LocalCluster(n_workers=16, threads_per_worker=1)) as c:
        results = c.map(extract_vcp, rad_files)
        results = c.gather(results)
    time = np.array([x[0] for x in results])
    vcp = np.array([x[1] for x in results])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(time, vcp)
    ax.set_ylabel('VCP')
    ax.set_xlabel('Time [UTC]')
    
    ds = xr.Dataset({'time': time, 'vcp': vcp})
    ds["vcp"].attrs["standard_name"] = "NEXRAD Volume Coverage Pattern"
    ds.to_netcdf(f"vcp_{radar}.nc")
