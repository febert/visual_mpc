import argparse
import os
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('dir', type=str, help='dir where to place files')
parser.add_argument('n', type=int, help='number of job-ids')
parser.add_argument('start', type=int, help='starting id')

args = parser.parse_args()
job_ids = []
dir = args.dir
for j in range(args.n):
    cmd = "cd {}; ngc result download {} &".format(dir, args.start + j)
    print(cmd)
    os.system(cmd)
