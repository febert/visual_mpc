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
         'step_duration' : 0.75,  #time each substep takes to execute
         'impedance_stiffness' : 150, #stiffness commanded to impedance controller
         'control_rate' : 1000,  #substep are taken at control_rate HZ
         'image_height' : 48,
         'image_width' : 64,
         'data_conf' : data_conf,  #controls cropping
         'adim' : 5,
         'sdim' : 5,
         'mode_rel' : np.array([True, True, True, True, False]),
         'targetpos_clip': [[0.42, -0.24, 0.184, -0.5 * np.pi , 0], [0.87, 0.22, 0.32, 0.5 * np.pi, 0.1]],
         'autograsp' : {'zthresh' :  0.15, 'touchthresh' : 0.0, 'reopen' : ''},   #15% of total height is zthresh,
         'file_to_record' : convert_to_record
         }

policy = {
    'type' : Randompolicy,
    'nactions' : 5,
    'repeat' : 3,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
    'initial_std_rot' : np.pi / 18,
    'initial_std_grasp' : 2
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 30000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'nshuffle' : 200
}
