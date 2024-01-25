import pyart
import cmweather
import glob
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs

from distributed import Client, LocalCluster, wait


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
    rad_gatefilter = pyart.filters.GateFilter(rad)
    rad_gatefilter.exclude_below('reflectivity', 0)
    rad_gatefilter = pyart.correct.despeckle_field(
        rad, 'velocity', gatefilter=rad_gatefilter)
    corrected_velocity = pyart.correct.dealias_region_based(
        rad, gatefilter=rad_gatefilter, centered=True)
    rad.add_field('corrected_velocity', corrected_velocity, replace_existing=True)
    fig, ax = plt.subplots(3, 2, figsize=(15, 15),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    disp = pyart.graph.RadarMapDisplay(rad)
    disp.plot_ppi('reflectivity', sweep=0, cmap='ChaseSpectral',
                      vmin=-20, vmax=80, ax=ax[0, 0])
    disp.plot_ppi('velocity', sweep=1, cmap='balance',
                      vmin=-60, vmax=60, ax=ax[1, 0])
    disp.plot_ppi('corrected_velocity', sweep=1, cmap='balance',
                      vmin=-60, vmax=60, ax=ax[0, 1])
    disp.plot_ppi('corrected_velocity', sweep=3,
                      cmap='balance',
                      vmin=-60, vmax=60, ax=ax[1, 1])
    disp.plot_ppi('corrected_velocity', sweep=5, cmap='balance',
                      vmin=-60, vmax=60, ax=ax[2, 0])
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim([-150, 150])
            ax[i, j].set_ylim([-150, 150])

    fig.savefig(os.path.join(out_plot_path, name + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    pyart.io.write_cfradial(os.path.join(out_proc_path, name + '.nc'), rad)
    del fig
    print("Quicklooks/Dealiasing for %s completed!" % file)

if __name__ == "__main__":
    file_list = glob.glob(rad_path + '/*')
    with Client(LocalCluster()) as c:
        results = c.map(make_quicklooks, file_list)
        wait(results)
