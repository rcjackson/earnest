import nexradaws
import sys
import os

out_path = '/eagle/CPOL/Earnest'

conn = nexradaws.NexradAwsInterface()
date_str = sys.argv[1]
rad_str = sys.argv[2]
out_path = os.path.join(out_path, rad_str)
print(date_str[5:6])
avail_scans = conn.get_avail_scans(
        date_str[0:4], date_str[4:6], date_str[6:8], rad_str)
print(avail_scans)
print(date_str)

results = conn.download(avail_scans, out_path)


