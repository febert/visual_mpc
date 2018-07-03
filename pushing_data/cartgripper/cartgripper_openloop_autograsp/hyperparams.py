""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
from python_visual_mpc.visual_mpc_core.envs.cartgripper_env.vanilla_env import VanillaCartgripperEnv
from python_visual_mpc.visual_mpc_core.infrastructure.utility.tfrecord_from_file import grasping_touch_file2record as convert_to_record


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'cartgripper_grasp.xml',
    'num_objects': 4,
    'object_mass': 0.1,
    'friction': 1.0,
    'mode_rel': np.array([True, True, True, True, False]),
    'finger_sensors': True,
}

agent = {
    'type': AgentMuJoCo,
    'env': (VanillaCartgripperEnv, env_params),
    'data_save_dir': BASE_DIR + '/train',
    'not_use_images':"",
    'adim':5,
    'sdim':5,
    'cameras':['maincam', 'leftcam'],
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'ztarget':0.13,
    'min_z_lift':0.05,
    'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record/',
    'discrete_gripper' : -1, #discretized gripper dimension,
    'lift_rejection_sample' : 15,
}

policy = {
    'type' : RandomPickPolicy,
    'nactions' : 10,
    'repeat' : 3,
    'no_action_bound' : False, 
    'initial_std': 0.02,   #std dev. in xy
    'initial_std_lift': 1.6,   #std dev. in xy
    'initial_std_rot' : np.pi / 18,
    'initial_std_grasp' : 2 
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
