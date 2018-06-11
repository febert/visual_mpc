""" creates dataset of start end configurations """


import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

folder_name = '/'.join(str.split(__file__, '/')[-2:-1])
current_dir = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
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
    'T': 17,
    'skip_first': 20,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 96,
    'image_width' : 128,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_initial_pos':'',
    'pos_disp_range':0.2,
    'ang_disp_range':np.pi/8,
    'arm_disp_range':0.2,
}


policy = {
    'type' : Randompolicy,
    'nactions': 4,
    'repeat': 1, # number of repeat for each action
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 1e-5,   #std dev. in xy
    # 'initial_std_grasp': 1e-5,   #std dev. in xy
}

config = {
    'current_dir':current_dir,
    # 'save_raw_images':'',
    'traj_per_file':128,
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'simulator':CollectGoalImageSim
}

