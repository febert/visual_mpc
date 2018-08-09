""" Hyperparameters for Robot Large Scale Data Collection (RLSDC) """

import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
import os
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'autograsp': {'reopen': True, 'zthresh': 0.15},
    'opencv_tracking': True,
    'video_save_dir': ''      # so long as this key is not None videos will correctly save in benchmark mode
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'robot_name' : 'sudri',
         'data_save_dir': BASE_DIR,
         'T' : 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height' : 48,
         'image_width' : 64,
         'benchmark_exp':'',
         'make_final_gif_pointoverlay': True,
         }

policy = {
    'type': Randompolicy,
    'nactions': 5,
    'repeat': 3,
    'no_action_bound': False,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
    'initial_std_rot': np.pi / 18,
    'initial_std_grasp': 1
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 30000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'nshuffle' : 200
}
