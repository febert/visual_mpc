import json
import argparse
import os

import pdb
parser = argparse.ArgumentParser(description='upload dataset')
parser.add_argument('datasetdir', type=str, help='relative path to script to dataset')

args = parser.parse_args()
dataset_name = str.split(args.datasetdir, '/')[-1]

cmd = "ngc dataset upload --source {0} --desc \"{1}\" {2} --ace nv-us-west-2".format(args.datasetdir, dataset_name, dataset_name)
print(cmd)
os.system(cmd)
