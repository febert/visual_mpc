import argparse
import os
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('dir', type=str, help='relative path to script to withing visual_mpc directory')

args = parser.parse_args()
job_ids = []
dir = args.dir
job_start_id = 41212
n = 10
for j in range(n):
    cmd = "cd {}; ngc result download {} &".format(dir, job_start_id + j)
    print(cmd)
    os.system(cmd)
