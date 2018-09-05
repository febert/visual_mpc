""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.autograsp_env import AutograspCartgripperEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'cartgripper_grasp.xml',
    'num_objects': 1,
    'object_mass': 0.1,
    'friction': 1.0,
    'finger_sensors': True,
    'autograsp': {'zthresh': -0.06, 'touchthresh': 0.0, 'reopen': True}
}

agent = {
    'type': GeneralAgent,
    'env': (AutograspCartgripperEnv, env_params),
    'data_save_dir': BASE_DIR + '/clipz/',
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 400,   #generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'discrete_gripper': -1, #discretized gripper dimension,
#    'rejection_sample': 5,
    'make_final_gif': '',
#    'master': 'sudeep@deepthought.banatao.berkeley.edu',
#    'master_datadir': '/raid/sudeep/sawyer_sim/autograsp_newphysics_3/'
}

policy = {
    'type' : Randompolicy,
    'nactions' : 10,
    'initial_std': 0.02,   #std dev. in xy
    'initial_std_lift': 1.6,   #std dev. in xy
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
