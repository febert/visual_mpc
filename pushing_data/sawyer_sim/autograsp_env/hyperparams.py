""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.autograsp_sawyer_mujoco_env import AutograspSawyerMujocoEnv


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'sawyer_grasp.xml',
    'num_objects': 4,
    'object_mass': 1,
#    'print_delta': True,
    'friction': 1,
    'substeps': 200,
    'autograsp': {'zthresh': 0.18, 'touchthresh': 0.0, 'reopen': True},
    'object_meshes': ['Bowl'],
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    # 'verbose_dir': '/home/sudeep/Desktop/mjc_debug/',
    'clean_xml': False
}

agent = {
    'type': GeneralAgent,
    'env': (AutograspSawyerMujocoEnv, env_params),
    'data_save_dir': BASE_DIR + '/clipz/',
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 400,   #generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'discrete_gripper': -1, #discretized gripper dimension,
#    'rejection_sample': 5,
#    'make_final_gif': '',
#    'master': 'sudeep@deepthought.banatao.berkeley.edu',
#    'master_datadir': '/raid/sudeep/sawyer_sim/autograsp_newphysics_3/'
}

policy = {
    'type': Randompolicy,
    'nactions': 10,
    # 'repeat': 3,
    # 'action_bound': True,
    # 'initial_std': 0.05,   #std dev. in xy
    # 'initial_std_lift': 0.15,   #std dev. in xy
    # 'initial_std_rot': np.pi / 18,
    # 'initial_std_grasp': 2,
    # 'max_z_delta': 0.05
}

config = {
    'traj_per_file': 16,
    'seperate_good': True,
    'current_dir': current_dir,
    'save_raw_images': True,
    'save_data': True,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
