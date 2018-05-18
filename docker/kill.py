import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('start', type=int, help='start id')
parser.add_argument('--num', type=int, help='number to kill', default=10)
args = parser.parse_args()
job_start_id = args.start
n = 10
for j in range(n):
    cmd = "ngc batch kill {} &".format(job_start_id + j)
    print(cmd)
    os.system(cmd)
