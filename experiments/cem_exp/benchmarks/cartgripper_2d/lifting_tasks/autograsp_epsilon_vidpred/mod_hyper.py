""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}

agent = {
    'type': BenchmarkAgent,
    'env': (CartgripperXZGrasp, env_params),
    'T': 60,
    'image_height' : 48,
    'image_width' : 64,
    'make_final_gif_pointoverlay': True,
    'record': BASE_DIR + '/record/',
    'start_goal_confs': os.environ['VMPC_DATA_DIR'] + '/cartgripper_xz_grasp/lifting_tasks',
    'current_dir': current_dir
}

policy = {
    'verbose':True,
    'type': CEM_Controller_Vidpred,
    'action_order': ['x', 'z', 'grasp'],
    'initial_std_lift': 0.5,  # std dev. in xy
    'rejection_sampling': False,
    'autograsp_epsilon': [-0.06, 0.2],
    'replan_interval': 15,
    'num_samples': [3200, 1600],
}

config = {
    'current_dir': current_dir,
    'save_data': False,
    'start_index':0,
    'end_index': 150,
    'agent': agent,
    'policy': policy,
}
