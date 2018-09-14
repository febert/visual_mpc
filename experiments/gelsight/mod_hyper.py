import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.algorithm.policy import DummyPolicy
from python_visual_mpc.visual_mpc_core.envs.gelsight.gelsight_env import GelsightEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'robot_name': 'gelsight'
}

agent = {
    'type': GeneralAgent,
    'env': (GelsightEnv, env_params),
    'T': 18,
    'image_height' : 48,
    'image_width' : 64,
    'image_channels': 3,
    'gen_xml': 1,   #generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'data_save_dir': BASE_DIR + '/train',
    'adim': 4,
    'sdim': 5,

}

policy = {
    'type': Randompolicy,
    'nactions': 6,
    'initial_std': 40
}


config = {
    'current_dir' : current_dir,
    'traj_per_file': 5,
    'save_data': True,
    'start_index': 0,
    'end_index': 50,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
}
