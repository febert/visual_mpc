""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division
import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.det_grasp_policy import DeterministicGraspPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'gtruth_planner': True,
    'data_collection': True,
    'sample_objectpos':'',
    'adim':5,
    'sdim':12,
    'xpos0': np.array([0., 0., 0.05, 0., 0., 0.]), #initialize state dimension to 5 zeros
    'dt': 0.05,
    'substeps': 50,  #6
    'T': 100,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_ballinitpos':'', #randomize x, y
    'custom_poscontroller':True,
    'ztarget':0.13,
    'drop_thresh':0.02,
    'make_final_gif':True,
    'record': BASE_DIR + '/record/'
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : DeterministicGraspPolicy,
    'nactions': 100,
    'iterations':1,
    'repeats': 5, # number of repeats for each action
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 20,   #std dev. in xy
    'debug_viewer':True,
    'num_samples':20,
    'best_to_take':5,
    # 'initial_std_grasp': 1e-5,   #std dev. in xy
}

config = {
    'traj_per_file':128,
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
