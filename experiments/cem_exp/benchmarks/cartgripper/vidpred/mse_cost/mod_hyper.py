""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xyz import CartgripperXYZEnv
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_sim import CEM_Controller_Sim
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.mse_full_image_controller import MSE_Full_Image_Controller


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
    'type': BenchmarkAgent,
    'env': (CartgripperXYZEnv, env_params),
    'data_save_dir': os.environ['VMPC_DATA_DIR'] + '/cartgripper/newenv/pushing_demo',
    'T': 35,
    'image_height' : 48,
    'image_width' : 64,
    'gen_xml': 1,   #generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'discrete_gripper': -1, #discretized gripper dimension,
    'make_final_gif': True,
    'start_goal_confs':os.environ['VMPC_DATA_DIR'] + '/cartgripper/newenv/startgoal_conf/train',
    'current_dir':current_dir,
    'ncam':1
}

policy = {
    'verbose':True,
    'type': MSE_Full_Image_Controller,
    'iterations':2,
    # 'num_samples':10, ##############
    # 'selection_frac':0.5,
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images':'',
    'start_index':0,
    'end_index': 50,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}