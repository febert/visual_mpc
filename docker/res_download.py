import argparse
import os
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('dir', type=str, help='relative path to script to withing visual_mpc directory')
parser.add_argument('n', type=int, help='relative path to script to withing visual_mpc directory')
parser.add_argument('start', type=int, help='relative path to script to withing visual_mpc directory')

args = parser.parse_args()
job_ids = []
dir = args.dir
n = 4
for j in range(args.n):
    cmd = "cd {}; ngc result download {} &".format(dir, args.start + j)
    print(cmd)
    os.system(cmd)
