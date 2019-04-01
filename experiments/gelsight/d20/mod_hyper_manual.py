import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.gelsight.gelsight_env import GelsightEnv, TBConfig
from python_visual_mpc.visual_mpc_core.algorithm.gelsight_manual_policy import GelsightManualPolicy


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
print(BASE_DIR)
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

VMPC_BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'home_pos': [5000, 5300, 940],
    'task': TBConfig.DICE,
    'robot_name': 'gelsight'
}

agent = {
    'type': GeneralAgent,
    'env': (GelsightEnv, env_params),
    'T': 35,
    'image_height' : 48,
    'image_width' : 64,
    'image_channels': 3,
    'record': BASE_DIR + '/record/',
    'data_save_dir': BASE_DIR + '/train/',
    'start_goal_confs': VMPC_BASE_DIR + '/', # No goals
    'current_dir': current_dir,
    'num_load_steps': 35,
    'no_flip': True
}

policy = {
    'type': GelsightManualPolicy,
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
