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
    'data_save_dir': '/mnt/sda1/pushing_data/cartgripper/grasping/' + folder_name + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':5,
    'sdim':12,
    'xpos0': np.array([0., 0., 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 2,
    'skip_first': 0, ########   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':5,   #generate xml every nth trajecotry
    'pos_disp_range':0.0,
    'ang_disp_range':0.0,
    'arm_disp_range':0.0,
    'cameras':['maincam', 'leftcam'],
    'lift_object':'',
    'arm_obj_initdist':0.0,
    'gen_new_goalpose':'',
    'arm_start_lifted':0.14,
}

policy = {
    'type' : lambda x, y: None,
}

config = {
    'save_raw_images':'',
    'save_data': True,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}

