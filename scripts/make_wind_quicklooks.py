import pyart
import cmweather
import glob
import matplotlib.pyplot as plt
import sys
import os
import cartopy.crs as ccrs

from distributed import Client, LocalCluster, wait


radar = sys.argv[1]
out_path = os.path.join('/lcrc/group/earthscience/rjackson/Earnest/quicklooks_wind', radar)
rad_path = os.path.join('/lcrc/group/earthscience/rjackson/Earnest/', radar)

if not os.path.exists(out_path):
    os.makedirs(out_path)

def make_quicklooks(file):    
    try:
        rad = pyart.io.read(file)
    except TypeError:
        return
    base, name = os.path.split(file)
    grid['spd'].isel(z=0).plot(cmap='coolwarm', vmin=0, vmax=60, ax=ax[0])
    pydda.vis.plot_horiz_xsection_quiver(grids, ax=ax[1], level=0)
    ax[1].set_title('Winds at 400 m')
    ax[0].set_ylabel('Y [m]')
    ax[0].set_xlabel('X [m]')
    ax[0].set_title(grid['time'].attrs["units"].split(" ")[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(quicklook_dir, name + '.wspd.png'), bbox_inches='tight', dpi=150)

    plt.close(fig)
    del fig
    print("Quicklooks for %s completed!" % file)

if __name__ == "__main__":
    file_list = glob.glob(rad_path + '/*')
    with Client(LocalCluster()) as c:
        results = c.map(make_quicklooks, file_list)
        wait(results)
