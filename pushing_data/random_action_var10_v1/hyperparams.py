""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

num_objects = 4

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/pushing2d.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/pushing2d.xml',
    'data_collection': True,
    'x0': np.array([0., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 15,
    'skip_first': 5,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 64,
    'image_width' : 64,
    'image_channels' : 3,
    'num_objects': num_objects,
    'novideo':''
}

policy = {
    'type' : Randompolicy,
    'initial_var': 10,
    'numactions': 5, # number of consecutive actions
    'repeats': 3, # number of repeats for each action
}

config = {
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'verbose_policy_trials': 0,
    'agent': agent,
    'gui_on': False,
    'policy': policy,
    'traj_per_file':10
}

# common['info'] = generate_experiment_info(config)
