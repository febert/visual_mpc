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

import python_visual_mpc
ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': os.environ['VMPC_DATA_DIR'] + '/cartgripper/'+ folder_name + '/train',
    'filename': ROOT_DIR + '/mjc_models/cartgripper_updown_whitefingers.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper_updown_whitefingers.xml',
    'data_collection': True,
    'adim':3,
    'sdim':6,
    'xpos0': np.array([0., 0., 0.]),
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
    'num_objects': 2,
    'novideo':'',
    'gen_xml':3,   #generate xml every nth trajecotry
    'ang_disp_range':np.pi/8,
    'arm_disp_range':0.2,
    'sample_objectpos':'',
    'object_object_mindist':0.35,
    'const_dist':0.2,
    'randomize_initial_pos':'',
    'first_last_noarm':''
}

policy = {
    'type' : lambda x, y: None,
}

config = {
    'save_raw_images':'',
    'save_data': True,
    'start_index':0,
    'end_index': 199,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}

