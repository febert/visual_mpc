import json
import argparse
import os

parser = argparse.ArgumentParser(description='upload dataset')
parser.add_argument('datasetdir', type=str, help='relative path to script to dataset')
parser.add_argument('--name', type=str, default="", help='name of dataset')

args = parser.parse_args()

if args.name is not "":
    dataset_name = args.name
else:
    dataset_name = str.split(args.datasetdir, '/')[-1]

cmd = "ngc dataset upload --source {0} --desc \"{1}\" {2} --ace nv-us-west-2".format(args.datasetdir, dataset_name, dataset_name)
print(cmd)
os.system(cmd)
