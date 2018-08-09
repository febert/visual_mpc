""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.vanilla_sawyer_env import VanillaSawyerEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': GeneralAgent,
    'env': (VanillaSawyerEnv, {'video_save_dir': '/home/sudeep/Desktop/'}),
    'data_save_dir': BASE_DIR,
    'not_use_images':"",
    'cameras':['maincam', 'leftcam'],
    'T': 15,
    'image_height' : 48,
    'image_width' : 64,
    'novideo':'',
    'ztarget':0.13,
    'min_z_lift':0.05,
    'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record/',
    'discrete_gripper' : -1, #discretized gripper dimension,
}

policy = {
    'type': Randompolicy,
    'nactions': 5,
    'repeat': 3,
    'no_action_bound': False,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
    'initial_std_rot': np.pi / 18,
    'initial_std_grasp': 1
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
