""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
current_dir = '/'.join(str.split(__file__, '/')[:-1])

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': os.environ['RESULT_DIR'] + '/' + folder_name + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':3,
    'sdim':6,
    'xpos0': np.array([0., 0., 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 15,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 4,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_initial_pos':'',
    'not_use_images':""
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeat': 3, # number of repeat for each action
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 1e-5,   #std dev. in xy
    # 'initial_std_grasp': 1e-5,   #std dev. in xy
}

config = {
    'traj_per_file': 128,
    'current_dir':current_dir,
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
