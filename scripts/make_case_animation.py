import glob
import sys
from nexrad_animation import animate_nexrad_kokx_kbox

month = sys.argv[1]
kokx_files = sorted(glob.glob(f"/projects/storm/rjackson/wfip3/nexrad/KOKX/*{month}*"))
kbox_files = sorted(glob.glob(f"/projects/storm/rjackson/wfip3/nexrad/KBOX/*{month}*"))

anim = animate_nexrad_kokx_kbox(
        kokx_files,
        kbox_files,
        cmap="ChaseSpectral",
        output_path=f"kokx_kbox{month}.gif",
    )
