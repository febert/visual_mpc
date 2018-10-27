import numpy as np
import os
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from python_visual_mpc.visual_mpc_core.algorithm.custom_samplers.folding_sampler import FoldingSampler
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.goal_image_controller import GoalImageController
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from python_visual_mpc.goal_classifier.classifier_wrapper import ClassifierDeploy
from python_visual_mpc.goal_classifier.models.conditioned_model import ConditionedGoalClassifier
import python_visual_mpc
CLASSIFIER_DIR = '/'.join(python_visual_mpc.__file__.split('/')[:-2])
CLASSIFIER_DIR = '{}/tensorflow_data/goal_classifier/towel_classifier/'.format(CLASSIFIER_DIR)


env_params = {
    'lower_bound_delta': [0, 0., -0.01, 265 * np.pi / 180 - np.pi/2, 0],
    'upper_bound_delta': [0, -0.15, -0.01, 0., 0],
    'normalize_actions': True,
    'gripper_joint_thresh': 0.999856,
    'rand_drop_reset': False,
    'video_save_dir':  '',
    'start_box': [1, 1, 0.5],
    'zthresh':0.05   # gripper only closes very close to ground
}

agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T' : 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'current_dir': current_dir,
         'ntask': 2,
         'register_gtruth': ['start', 'goal']
         }

policy = {
    'verbose':True,
    'type': GoalImageController,
    'replan_interval': 15,
    'num_samples': [1200, 1200],
    'custom_sampler': FoldingSampler,
    'selection_frac': 0.05,
    'predictor_propagation': True,   # use the model get the designated pixel for the next step!
    'initial_std': 0.005,
    'initial_std_lift': 0.05,  # std dev. in xy
    'extra_score_functions': [ClassifierDeploy({'model': ConditionedGoalClassifier},
                                               '/home/annie/Downloads/25_10_2018/model99999')],
    'pixel_score_weight': 0,
    # 'extra_score_weight': 0
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
