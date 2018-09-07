import numpy as np
import os
from python_visual_mpc.visual_mpc_core.agent.benchmarking_agent import BenchmarkAgent
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.autograsp_sawyer_env import AutograspSawyerEnv
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    'autograsp': {'reopen': True, 'zthresh': 0.15},
    'video_save_dir': ''
}


agent = {'type' : BenchmarkAgent,
         'env': (AutograspSawyerEnv, env_params),
         'data_save_dir': BASE_DIR,
         'T' : 50,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'image_height': 48,
         'image_width': 64,
         'benchmark_exp':'',
         'current_dir': current_dir,
         'ntask': 2
         }

policy = {
    'verbose':'',
    'verbose_every_itr':"",
    'type': CEM_Controller_Vidpred,
    'iterations': 3,
    'nactions': 5,
    'repeat': 3,
    'no_action_bound': False,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
    'initial_std_rot': np.pi / 18,
    'finalweight':10,
    'ncam': 2,
    'action_cost_factor': 0,
    'replan_interval': 3,
    'reuse_mean': '',
    'reuse_action_as_mean': "",
    'reduce_std_dev': 0.2,  # reduce standard dev in later timesteps when reusing action
    'num_samples': [400, 200],
    'selection_frac': 0.05,
    'predictor_propagation': '',   # use the model get the designated pixel for the next step!
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
