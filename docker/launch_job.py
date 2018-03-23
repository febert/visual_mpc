import json
import argparse
import os

import pdb
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('run_dir', type=str, help='relative directory to withing visual_mpc from where to run the script')
parser.add_argument('hyper', type=str, help='relative path to hyperparams file', default=10)

args = parser.parse_args()
run_dir = '/'.join(str.split(args.run_dir, '/')[1:-1])

hyper = '/'.join(str.split(args.hyper, '/')[1:])

command = "python " + str.split(args.run_dir, '/')[-1] + " --hyper ../../" + hyper

print("using command ", command)

data = {}
data['dockerImageName'] = "ucb_rail8888/tf1.4_gpu:based_nvidia"
data["aceName"] = "nv-us-west-2"
data["name"] = str.split(command, '/')[-2]
data["command"] =\
"cd /result && tensorboard --logdir . & \
 export VMPC_DATA_DIR=/mnt/pushing_data;\
 export MUJOCO_PY_MJKEY_PATH=/workspace/visual_mpc/mujoco/mjpro131/mjkey.txt;\
 export MUJOCO_PY_MJPRO_PATH=/workspace/visual_mpc/mujoco/mjpro131; \
 export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;\
 cd /workspace/visual_mpc/{0};\
 {1} --docker".format(run_dir, command)

data["datasetMounts"] = [{"containerMountPoint": "/mnt/pushing_data", "id": 8350}]
data["resultContainerMountPoint"] = "/result"
data["aceInstance"] = "ngcv1"
data["publishedContainerPorts"] = [6006] #for tensorboard

with open('autogen.json', 'w') as outfile:
    json.dump(data, outfile)

os.system("ngc batch run -f autogen.json")
