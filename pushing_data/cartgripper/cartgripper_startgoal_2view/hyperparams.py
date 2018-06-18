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

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': os.environ['VMPC_DATA_DIR'] + '/cartgripper/' + folder_name + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_updown_2cam.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_updown_2cam.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':3,
    'sdim':6,
    'xpos0': np.array([0., 0., 0.]),
    'object_mass':0.01,
    'friction':1.5,
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 2,
    'skip_first': 20,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':5,   #generate xml every nth trajecotry
    'randomize_initial_pos':'',
    'pos_disp_range':0.5,
    'ang_disp_range':np.pi/8,
    'arm_disp_range':0.2,
    'cameras':['maincam', 'leftcam']
}

policy = {
    'type' : lambda x, y, z: None,
}

config = {
    'save_raw_images':'',
    'save_data': True,
    'start_index':0,
    'end_index': 49,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}

