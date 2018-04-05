import json
import argparse
import os

import re

import pdb
parser = argparse.ArgumentParser(description='write json configuration for ngc')
parser.add_argument('run_dir', type=str, help='relative path to script to withing visual_mpc directory')
parser.add_argument('--hyper', type=str, help='relative path to hyperparams file', default="")
parser.add_argument('--int', default='False', type=str, help='interactive')
parser.add_argument('--arg', default='', type=str, help='additional arguments')

args = parser.parse_args()
run_dir = '/'.join(str.split(args.run_dir, '/')[1:-1])

hyper = '/'.join(str.split(args.hyper, '/')[1:])

script_name = str.split(args.run_dir, '/')[-1]

data = {}
data["aceName"] = "nv-us-west-2"
data["command"] =\
"cd /result && tensorboard --logdir . & \
 export VMPC_DATA_DIR=/mnt/pushing_data;\
 export TEN_DATA=/mnt/tensorflow_data;\
 export RESULT_DIR=/result;\
 export NO_ROS='';\
 export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;\
 cd /workspace/visual_mpc/{0};".format(run_dir)

data['dockerImageName'] = "ucb_rail8888/tf_mj1.5:latest"

if 'benchmarks' in script_name or 'parallel_data_collection' in script_name:  #running benchmark...
    data["datasetMounts"] = [{"containerMountPoint": "/mnt/tensorflow_data/sim/mj_pos_ctrl_appflow", "id": 8906},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/appflow_nogenpix", "id": 8933},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/cartgripper_flowonly", "id": 8952},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/mj_pos_ctrl", "id": 8930},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl", "id": 8948},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl_updown_rot_sact", "id": 8951},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl_updown_sact", "id": 8950},
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper_startgoal_short", "id": 8949},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper_startgoal_masks", "id": 8914}]  # mj_pos_ctrl_appflow
    data["aceInstance"] = "ngcv8"
    command = "python " + script_name + " {}".format(args.arg)

    data["name"] = '-'.join(re.compile('\w+').findall(args.arg))
else:
    data["aceInstance"] = "ngcv1"
    data["datasetMounts"] = [{"containerMountPoint": "/mnt/pushing_data/cartgripper", "id": 8350},  # cartgripper
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper_mj1.5", "id": 8974},
                             {"containerMountPoint": "/mnt/pushing_data/mj_pos_noreplan_fast_tfrec", "id": 8807},  #mj_pos_noreplan_fast_tfrec    | gtruth mujoco planning pushing
                             {"containerMountPoint": "/mnt/pushing_data/mj_pos_noreplan_fast_tfrec_fewdata", "id": 8972},  #mj_pos_noreplan_fast_tfrec    | gtruth mujoco planning pushing
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper_updown_sact", "id": 8950},
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper_updown_rot_sact", "id": 8951}]

    command = "python " + script_name + " --hyper ../../" + hyper
    data["name"] = str.split(command, '/')[-2]

if args.int == 'True':
    command = "/bin/sleep 3600"

data["command"] += command

data["resultContainerMountPoint"] = "/result"
data["publishedContainerPorts"] = [6006] #for tensorboard

with open('autogen.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

os.system("ngc batch run -f autogen.json")
