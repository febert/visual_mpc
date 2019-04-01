import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy, RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.envs.gelsight.gelsight_env import GelsightEnv
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.mse_full_image_controller import MSE_Full_Image_Controller
import datetime

ctimestr = 'mpc-' + datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/exp_' + ctimestr
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

VMPC_BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    'home_pos': [2650, 6040, 1050]
}

agent = {
    'type': BenchmarkAgent,
    'env': (GelsightEnv, env_params),
    'T': 20,
    'image_height' : 48,
    'image_width' : 64,
    'image_channels': 3,
    'record': BASE_DIR + '/record/',
    'data_save_dir': BASE_DIR + '/trajectories',
    'adim': 3,
    'sdim': 3,
    'start_goal_confs': VMPC_BASE_DIR + '/experiments/gelsight/bench_goal_images_reshaped',
    'current_dir': current_dir,
    'num_load_steps': 18,
    '_bench_save': current_dir + '/bench',
    'make_final_gif': True,
    'no_flip': True,
    'benchmark_exp':''
}


policy = {
    'type': MSE_Full_Image_Controller,
    'initial_std': 1,   #std dev. in xy
    'initial_std_lift': 1,
    'verbose': True,
    'add_zero_actions':True,
    #'finalweight': 1
    #'verbose_every_itr': True,
    #'iterations': 3  
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
