""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.autograsp_sawyer_mujoco_env import AutograspSawyerMujocoEnv

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.register_gtruth_controller import Register_Gtruth_Controller

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'filename': 'sawyer_grasp.xml',
    'num_objects': 1,
    'object_mass': 0.1,
    'friction': 1,
    'finger_sensors': True,
    'substeps': 100,
     'autograsp': {'zthresh': 0.18, 'touchthresh': 0.0, 'reopen': True},
    'object_meshes': ['Bowl'],
    'ncam':2,
}

agent = {
    'type': BenchmarkAgent,
    'env': (AutograspSawyerMujocoEnv, env_params),
    'T': 30,
    'image_height' : 48,
    'image_width' : 64,
    'novideo':'',
    'gen_xml':1,   #generate xml every nth trajecotry
    'ztarget':0.13,
    'min_z_lift':0.05,
    'make_final_gif': True,
    'discrete_gripper': -1, #discretized gripper dimension,
    'start_goal_confs':os.environ['VMPC_DATA_DIR'] + '/sawyer_sim/startgoal_conf/bowl_arm_disp/train',
    'current_dir': current_dir,
}

policy = {
    'verbose':'',
    # 'verbose_every_itr':"",
    'type': Register_Gtruth_Controller,
    'iterations': 3,
    'nactions': 5,
    'repeat': 3,
    'no_action_bound': False,
    'initial_std': 0.05,   #std dev. in xy
    'initial_std_lift': 0.15,   #std dev. in xy
    'initial_std_rot': np.pi / 18,
    'finalweight':10,
    'register_gtruth':['start','goal'],
    'trade_off_reg':'',
}

config = {
    'traj_per_file': 128,
    'current_dir': current_dir,
    'save_data': False,
    'start_index':0,
    'end_index': 50,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
