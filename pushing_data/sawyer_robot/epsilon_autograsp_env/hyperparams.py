""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import RandomEpsilonAG
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.vanilla_sawyer_env import VanillaSawyerEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': GeneralAgent,
    'env': (VanillaSawyerEnv, {}),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 240,
    'image_width' : 320,
    # 'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record/',
}

policy = {
    'type': RandomEpsilonAG,
    'nactions': 10,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
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
