""" creates dataset of start end configurations """


import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

current_dir = '/'.join(str.split(__file__, '/')[:-1])
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3


import python_visual_mpc
folder_name = '/'.join(str.split(__file__, '/')[-2:-1])
from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim

BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': os.environ['VMPC_DATA_DIR'] + '/' + folder_name + '/train',
    'filename': BASE_DIR+'/mjc_models/cartgripper.xml',
    'filename_nomarkers': BASE_DIR+'/mjc_models/cartgripper.xml',
    'data_collection': True,
    'sample_objectpos':'',
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
    'num_objects': 1,
    'novideo':'',
    'gen_xml':5,   #generate xml every nth trajecotry
    'init_arm_near_obj':1e-1,
    'pos_disp_range':0.1,
    'ang_disp_range':np.pi/16,
    'arm_disp_range':0.1,
    'goal_mask':'',
}


policy = {
    'type' : lambda x, y: None,
}

config = {
    'current_dir':current_dir,
    'save_raw_images':'',
    'save_data': True,
    'start_index':0,
    'end_index': 500,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'simulator':CollectGoalImageSim
}

