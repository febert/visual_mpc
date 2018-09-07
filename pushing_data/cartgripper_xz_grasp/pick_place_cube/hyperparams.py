from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.demonstration_envs.pick_place_env import CartgripperXZGPickPlace
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.algorithm.demonstration_policies.demo_policy import DemoPolicy
from python_visual_mpc.visual_mpc_core.algorithm.demonstration_policies.util.stage_graphs import PickAndPlace
import os
import numpy as np


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}

agent = {
    'type': GeneralAgent,
    'env': (CartgripperXZGPickPlace, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height': 48,
    'image_width': 64,
    'gen_xml': 1,  # generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'rejection_sample': 50,
}


sub_policy_hparams = [
    {'active_dims': [True, False, True, False, True], 'xyz_bias': [0., 0., 0.15], 'max_rot': 0., 'tolerance': 0.01},
    {'active_dims': [True, False, True, False, True], 'truncate_action': [0.01, 0.01, 0.3, np.pi / 8, 1.]},
    {'active_dims': [True, False, True, False, True], 'xyz_bias': [0., 0., 0.15], 'max_rot': 0., 'tolerance': 0.01,
     'bounds': [[-0.4, 0., 0.1], [0.4, 0., 0.15]]},
    {'active_dims': [True, False, True, False, True], 'truncate_action': [0.01, 0.01, 0.1, np.pi / 8, 1.]},
    {'active_dims': [True, False, True, False, True], 'x_noise': [0., 0.1], 'z_noise': [0., 0.1], 'go_up_first': True}
]

policy = {
    'type': DemoPolicy,
    'stage_graph': PickAndPlace,
    'stage_params': (sub_policy_hparams, [0, 10, 20, 5])
}

config = {
    'traj_per_file': 16,
    'current_dir': current_dir,
#    'save_raw_images': True,
    'save_only_good': True,
    'seperate_good': True,
    'start_index':0,
    'end_index': 70000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}