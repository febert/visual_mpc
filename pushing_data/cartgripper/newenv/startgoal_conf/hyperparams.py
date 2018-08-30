import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.create_configs_agent import CreateConfigAgent
from python_visual_mpc.visual_mpc_core.algorithm.policy import DummyPolicy
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xyz import CartgripperXYZEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'num_objects': 1,
    'object_mass': 0.1,
    'friction': 1.0,
}

agent = {
    'type': CreateConfigAgent,
    'env': (CartgripperXYZEnv, env_params),
    'T': 1,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 1,   #generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'discrete_gripper': -1, #discretized gripper dimension,
    'data_save_dir':os.environ['VMPC_DATA_DIR'] + '/cartgripper/newenv/startgoal_conf',
}

policy = {
    'type': DummyPolicy
}


config = {
    'save_raw_images':'',
    'seperate_good': True,
    'current_dir' : current_dir,
    'save_data': True,
    'start_index':0,
    'end_index': 50,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'sequence_length':2
}
