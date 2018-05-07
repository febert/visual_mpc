import json
import argparse
import pdb
import os
import re
import pdb

def launch_job_func(run_script, hyper, arg, interactive=False, name='', ngpu=8, nsplit=None, isplit=None):

    data = {}
    data["aceName"] = "nv-us-west-2"
    data["command"] = \
    "cd /result && tensorboard --logdir . & \
     export VMPC_DATA_DIR=/mnt/pushing_data;\
     export TEN_DATA=/mnt/tensorflow_data;\
     export ALEX_DATA=/mnt/pretrained_models;\
     export RESULT_DIR=/result;\
     export NO_ROS='';\
     export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;\
     cd /workspace/visual_mpc/docker;"

    data['dockerImageName'] = "ucb_rail8888/tf_mj1.5:latest"

    script_name = run_script

    data["datasetMounts"] = [{"containerMountPoint": "/mnt/tensorflow_data/sim/mj_pos_ctrl_appflow", "id": 8906},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/appflow_nogenpix", "id": 8933},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/onpolicy/updown_sact_onpolonly", "id": 9225},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/onpolicy/updown_sact_comb", "id": 9224},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/appflow_nogenpix_mj1.5", "id": 9006},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/cartgripper_flowonly", "id": 9007},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/mj_pos_ctrl", "id": 8930},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl", "id": 8948},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl_updown_rot_sact", "id": 8951},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/pos_ctrl_updown_sact", "id": 8950},
                             {"containerMountPoint": "/mnt/tensorflow_data/sim/onpolicy/updown_sact_onpolonly_iter2", "id": 9525},
                             {"containerMountPoint": "/mnt/tensorflow_data/gdn/startgoal_shad", "id": 9087},
                             {"containerMountPoint": "/mnt/tensorflow_data/gdn/96x128/cartgripper_tdac_flowpenal", "id": 9287},
                             {"containerMountPoint": "/mnt/pretrained_models/bair_action_free/model.savp.transformation.flow.last_frames.2.generate_scratch_image.false.batch_size.16", "id": 9161},
                             {"containerMountPoint": "/mnt/pretrained_models/bair_action_free/model.multi_savp.ngf.64.shared_views.true.num_views.2.tv_weight.0.001.transformation.flow.last_frames.2.generate_scratch_image.false.batch_size.16", "id": 9223},
                             {"containerMountPoint": "/mnt/pretrained_models/bair_action_free/model.multi_savp.num_views.2.tv_weight.0.001.transformation.flow.last_frames.2.generate_scratch_image.false.batch_size.16", "id": 9387},
                             {"containerMountPoint": "/mnt/pretrained_models/lift_push_last_subsequence/vae/model.savp.tv_weight.0.001.transformation.flow.last_frames.2.generate_scratch_image.false.batch_size.16", "id":9534},
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_startgoal_masks6e4", "id": 9138},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_const_dist", "id": 9259},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_startgoal_short", "id": 8949},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_startgoal_2view", "id": 9222},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_startgoal_masks", "id": 8914},  # mj_pos_ctrl_appflow
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper", "id": 8350},  # cartgripper
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/grasping/cartgripper_lift_benchmark", "id": 9538},  # cartgripper
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_mj1.5", "id": 8974},
                             {"containerMountPoint": "/mnt/pushing_data/onpolicy/mj_pos_noreplan_fast_tfrec", "id": 8807},  #mj_pos_noreplan_fast_tfrec    | gtruth mujoco planning pushing
                             {"containerMountPoint": "/mnt/pushing_data/onpolicy/mj_pos_noreplan_fast_tfrec_fewdata", "id": 8972},
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_updown_sact", "id": 8931},
                             {"containerMountPoint": "/mnt/pushing_data/cartgripper/cartgripper_updown_rot_sact", "id": 8951},
                             {"containerMountPoint": "/mnt/pushing_data/onpolicy/updown_sact_bounded_disc", "id": 9363}
                             ]

    data["aceInstance"] = "ngcv{}".format(ngpu)

    if interactive == 'True':
        command = "/bin/sleep 360000"
        data["name"] = 'int' + name
    else:
        if 'benchmarks' in script_name or 'parallel_data_collection' in script_name:  #running benchmark...
            if nsplit is not None:
                split = '--nsplit {} --isplit {}'.format(nsplit, isplit)
            else: split = ''
            command = "python " + run_script + " " + hyper + " {} {}".format(arg, split)
            expname = hyper.partition('benchmarks')[-1]
            data["name"] = '-'.join(re.compile('\w+').findall(expname + arg))
        else:
            command = "python " + run_script + " --hyper ../../" + hyper
            data["name"] = str.split(command, '/')[-2]

    data["command"] += command

    data["resultContainerMountPoint"] = "/result"
    data["publishedContainerPorts"] = [6006] #for tensorboard

    with open('autogen.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print('command', command)

    os.system("ngc batch run -f autogen.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='write json configuration for ngc')
    parser.add_argument('run_script', type=str, help='relative path to the script to launch', default="")
    parser.add_argument('hyper', type=str, help='relative path to hyperparams file', default="")
    parser.add_argument('--int', default='False', type=str, help='interactive')
    parser.add_argument('--arg', default='', type=str, help='additional arguments')
    parser.add_argument('--name', default='', type=str, help='additional arguments')
    parser.add_argument('--ngpu', default=8, type=int, help='number of gpus')
    args = parser.parse_args()

    launch_job_func(args.run_script, args.hyper, args.arg, args.int, args.name, args.ngpu)

