""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division
import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_files_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':3,
    'sdim':3,
    'xpos0': np.array([0., 0., 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 15,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 64,
    'image_width' : 64,
    'image_channels' : 3,
    'num_objects': 4,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_ballinitpos':'',
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeats': 3, # number of repeats for each action
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 1e-5,   #std dev. in xy
    # 'initial_std_grasp': 1e-5,   #std dev. in xy
}

config = {
    'traj_per_file':128,
    'save_data': False,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
