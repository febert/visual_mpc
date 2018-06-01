""" creates dataset of start end configurations """


import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

current_dir = '/'.join(str.split(__file__, '/')[:-1])
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':5,
    'sdim':12,
    'dt': 0.05,
    'substeps': 10,  #6
    'T': 2,
    'skip_first': 0,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':5,   #generate xml every nth trajecotry
    # 'randomize_initial_pos':'',
    'pos_disp_range':0.,
    'ang_disp_range':0.,
    'arm_disp_range':0.,
    'lift_object':'',
    'arm_obj_initdist':0.05,
    'arm_start_lifted':0.14,
    'posmode':''
}

policy = {
    'type' : lambda x, y: None,
}

config = {
    'current_dir':current_dir,
    'save_raw_images':'',
    'save_data': True,
    'start_index':0,
    'end_index': 49,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}

