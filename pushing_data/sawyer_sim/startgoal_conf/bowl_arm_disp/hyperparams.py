import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.create_configs_agent import CreateConfigAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.vanilla_env import VanillaSawyerMujocoEnv
from python_visual_mpc.visual_mpc_core.algorithm.policy import DummyPolicy


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'sawyer_grasp.xml',
    'num_objects': 1,
    'object_mass': 0.1,
    'friction': 1.,
    'substeps': 100,
    'object_meshes': ['Bowl','ElephantBowl','GlassBowl','LotusBowl01','RuggedBowl','ServingBowl'],
}

agent = {
    'type': CreateConfigAgent,
    'env': (VanillaSawyerMujocoEnv, env_params),
    'data_save_dir': BASE_DIR,
    'not_use_images':"",
    'cameras':['maincam', 'leftcam'],
    'T': 1,
    'image_height' : 48,
    'image_width' : 64,
    'novideo':'',
    'gen_xml':1,   #generate xml every nth trajecotry
    'ztarget':0.13,
    'record': BASE_DIR + '/record/',
    'rejection_sample': 10,
    'data_save_dir':os.environ['VMPC_DATA_DIR'] + '/sawyer_sim/startgoal_conf/bowl_arm_disp',
}

policy = {
    'type': DummyPolicy
}


config = {
    'traj_per_file': 16,
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
