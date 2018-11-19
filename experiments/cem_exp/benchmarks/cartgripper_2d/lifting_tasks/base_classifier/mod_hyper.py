""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.goal_image_controller import GoalImageController
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.goal_classifier.classifier_wrapper import ClassifierDeploy


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
PROJ_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
print(PROJ_DIR)

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}

agent = {
    'type': BenchmarkAgent,
    'env': (CartgripperXZGrasp, env_params),
    'T': 45,
    'image_height' : 48,
    'image_width' : 64,
    'make_final_gif_pointoverlay': True,
    'record': BASE_DIR + '/record/',
    'start_goal_confs': os.environ['VMPC_DATA_DIR'] + '/cartgripper_xz_grasp/lifting_tasks',
    'current_dir': current_dir
}

policy = {
    'verbose':True,
    'type': GoalImageController,
    'action_order': ['x', 'z', 'grasp'],
    'initial_std_lift': 0.5,  # std dev. in xy
    'rejection_sampling': False,
    'replan_interval': 10,
    'num_samples': [800, 400],
    'pixel_score_weight': 0,
    'extra_score_functions': [(ClassifierDeploy, {'conf': '{}/tensorflow_data/goal_classifier/cartgripper_xz_vanilla/conf.py'.format(PROJ_DIR),
                                                  'checkpoint_path': '{}/tensorflow_data/goal_classifier/cartgripper_xz_vanilla/base_model/model-75000'.format(PROJ_DIR),
                                                  'device_id': None})]   # fill in default
}

config = {
    'current_dir': current_dir,
    'save_data': False,
    'start_index':0,
    'end_index': 150,
    'agent': agent,
    'policy': policy,
}
