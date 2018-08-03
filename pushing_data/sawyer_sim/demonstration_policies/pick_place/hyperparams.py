""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.demonstration_policies.demo_policy import DemoPolicy
from python_visual_mpc.visual_mpc_core.algorithm.demonstration_policies.util.stage_graphs import PickAndPlace
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.vanilla_env import PickPlaceDemo

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'sawyer_grasp.xml',
    'num_objects': 1,
    'object_mass': 0.1,
    'friction': 1.0,
    'finger_sensors': True,
    'object_meshes': ['LotusBowl01']
}

agent = {
    'type': GeneralAgent,
    'env': (PickPlaceDemo, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record/',
}

policy = {
    'type': DemoPolicy,
    'stage_graph': PickAndPlace
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
