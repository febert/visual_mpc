""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division
import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.det_grasp_policy import DeterministicGraspPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy

current_dir = '/'.join(str.split(__file__, '/')[:-1])
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': os.environ['RESULT_DIR'] + '/' + folder_name + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_updown.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_updown.xml',
    'not_use_images':"",
    'sample_objectpos':'',
    'adim':3,
    'sdim':6,
    'xpos0': np.array([0., 0., 0.1]), #initialize state dimension to 5 zeros
    'dt': 0.05,
    'substeps': 200,  #6
    'T': 15,
    'skip_first': 2,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 4,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_initial_pos':'', #randomize x, y
    'posmode':"",
    'targetpos_clip':[[-0.45, -0.45, -0.08], [0.45, 0.45, 0.15]],
    'discrete_adim':[2],
    # 'make_final_gif':'',
    'record':current_dir + '/verbose',
    'get_curr_mask':'',
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeat': 3,               # number of repeat for each action
    'initial_std': 0.08,        # std dev. in xy
    'initial_std_lift': 2.5, #0.1,
}

config = {
    'traj_per_file': 1,
    'current_dir':current_dir,
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
