""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import DummyPolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.human_demo_env import HumanDemoEnv

if 'VMPC_DATA_DIR' in os.environ:
    BASE_DIR = os.path.join(os.environ['VMPC_DATA_DIR'], 'towel_pick_demos/')
else:
    BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'lower_bound_delta': [0, 0., -0.01, 265 * np.pi / 180 - np.pi/2, 0],
    'upper_bound_delta': [0, -0.15, -0.01, 0., 0],
    'rand_drop_reset': False,
    'cleanup_rate': -1
}

agent = {
    'type': GeneralAgent,
    'env': (HumanDemoEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 0,
    'image_height' : 240,
    'image_width' : 320
}

policy = {
    'type': DummyPolicy
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
    'ngroup': 20
}
