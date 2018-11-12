import numpy as np
import os
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from python_visual_mpc.visual_mpc_core.algorithm.hardcoded_pick_place import HardcodedPickPlace

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'lower_bound_delta': [0, 0., -0.01, 265 * np.pi / 180 - np.pi/2, 0],
    'upper_bound_delta': [0, -0.15, -0.01, 0., 0],
    'normalize_actions': True,
    'gripper_joint_thresh': 0.999856,
    'rand_drop_reset': False,
    'start_box': [1, 1, 0.5],
    'video_save_dir':  '/home/sudeep/Desktop/exp_rec/',
    'zthresh':0.05   # gripper only closes very close to ground
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'point_space_width': 640,
         'T' : 30,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'ntask': 2
         }

policy = {
    'type': HardcodedPickPlace
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
