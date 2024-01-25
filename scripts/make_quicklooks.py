import pyart
import cmweather
import glob
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs

from distributed import Client, LocalCluster, wait


radar = sys.argv[1]
out_path = os.path.join('/lcrc/group/earthscience/rjackson/Earnest/quicklooks', radar)
rad_path = os.path.join('/lcrc/group/earthscience/rjackson/Earnest/', radar)

if not os.path.exists(out_path):
    os.makedirs(out_path)

def make_quicklooks(file):    
    try:
        rad = pyart.io.read(file)
    except TypeError:
        return
    base, name = os.path.split(file)
    fig, ax = plt.subplots(2, 2, figsize=(15, 15),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    disp = pyart.graph.RadarMapDisplay(rad)
    rad.info()
    disp.plot_ppi_map('reflectivity', sweep=0, cmap='ChaseSpectral',
                      vmin=-20, vmax=80, ax=ax[0, 0])
    disp.plot_ppi_map('velocity', sweep=1, cmap='balance',
                      vmin=-50, vmax=50, ax=ax[1, 0])
    disp.plot_ppi_map('differential_phase', sweep=0, cmap='ChaseSpectral',
                      vmin=0, vmax=10, ax=ax[0, 1])
    disp.plot_ppi_map('cross_correlation_ratio',
                      sweep=0, cmap='ChaseSpectral',
                      vmin=0, vmax=1, ax=ax[1, 1])
    for i in range(0, 2):
        for j in range(2):
            ax[i, j].set_xlim([-97, -90])
            ax[i, j].set_ylim([40, 44])
    fig.savefig(os.path.join(out_path, name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    del fig
    print("Quicklooks for %s completed!" % file)

if __name__ == "__main__":
    file_list = glob.glob(rad_path + '/*')
    with Client(LocalCluster()) as c:
        results = c.map(make_quicklooks, file_list)
        wait(results)
