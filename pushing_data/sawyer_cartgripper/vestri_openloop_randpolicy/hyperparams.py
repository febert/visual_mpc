""" Hyperparameters for Robot Large Scale Data Collection (RLSDC) """

import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
import os
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.agent.agent_robot import AgentSawyer
from python_visual_mpc.visual_mpc_core.infrastructure.utility.tfrecord_from_file import grasping_sawyer_file2record as convert_to_record
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

data_conf = {'left_cam' : {'crop_bot' : 70, 'crop_left' : 150, 'crop_right' : 100},
             'front_cam': {'crop_bot' : 70, 'crop_left' : 90, 'crop_right' : 160}}
agent = {'type' : AgentSawyer,
         'data_save_dir': BASE_DIR + '/train',
         'T' : 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'substeps' : 200,  #number of interpolated substeps per command
         'control_rate' : 1000,  #substep are taken at control_rate HZ
         'image_height' : 48,
         'image_width' : 64,
         'data_conf' : data_conf,  #controls cropping
         'adim' : 5,
         'sdim' : 5,
         'mode_rel' : np.array([True, True, True, True, False]),
         'discrete_gripper': -1,  # discretized gripper dimension,
         'targetpos_clip':[[0.42, -0.24, 0.184, -0.5 * np.pi , 0], [0.87, 0.22, 0.4, 0.5 * np.pi, 0.1]],
         'autograsp' : '',
         'autograsp_thresh' : 0.22,
         'file_to_record' : convert_to_record
         }

policy = {
    'type' : Randompolicy,
    'nactions' : 5,
    'repeat' : 3,
    'no_action_bound' : False,
    'initial_std': 0.1,   #std dev. in xy
    'initial_std_lift': 0.1,   #std dev. in z
    'initial_std_rot' : np.pi / 18,
    'initial_std_grasp' : 2
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
