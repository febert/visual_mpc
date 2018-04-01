import json
import argparse
import os

import pdb
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('run_dir', type=str, help='relative path to script to withing visual_mpc directory')
parser.add_argument('hyper', type=str, help='relative path to hyperparams file', default=10)
parser.add_argument('int', default='False', type=str, help='interactive')

args = parser.parse_args()
run_dir = '/'.join(str.split(args.run_dir, '/')[1:-1])

hyper = '/'.join(str.split(args.hyper, '/')[1:])

script_name = str.split(args.run_dir, '/')[-1]

if args.int == 'True':
    command = "/bin/bash"
else:
    command = "python " + script_name + " --hyper ../../" + hyper

print("using command ", command)

data = {}
data["aceName"] = "nv-us-west-2"
data["name"] = str.split(command, '/')[-2]
data["command"] =\
"cd /result && tensorboard --logdir . & \
 export VMPC_DATA_DIR=/mnt/pushing_data;\
 export TEN_DATA=/mnt/tensorflow_data;\
 export RESULT_DIR=/result;\
 export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;\
 cd /workspace/visual_mpc/{0};\
 {1} --docker".format(run_dir, command)

if 'benchmarks' or 'parallel_data_collection' in script_name:  #running benchmark...
    data["datasetMounts"] = [{"containerMountPoint": "/mnt/tensorflow_data/sim", "id": 8906}]  # mj_pos_ctrl_appflow
    data['dockerImageName'] = "ucb_rail8888/tf_mj1.5:latest"
else:
    data['dockerImageName'] = "ucb_rail8888/tf1.4_gpu:based_nvidia"
    data["datasetMounts"] = [{"containerMountPoint": "/mnt/pushing_data", "id": 8350},  # cartgripper
                             {"containerMountPoint": "/mnt/pushing_data", "id": 8807}]  #mj_pos_noreplan_fast_tfrec    | gtruth mujoco planning pushing


data["resultContainerMountPoint"] = "/result"
data["aceInstance"] = "ngcv1"
data["publishedContainerPorts"] = [6006] #for tensorboard

with open('autogen.json', 'w') as outfile:
    json.dump(data, outfile)

os.system("ngc batch run -f autogen.json")
