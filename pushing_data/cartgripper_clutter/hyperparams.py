""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.det_grasp_policy import CluteredGraspPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'not_use_images':"",
    'visible_viewer':False,
    'sample_objectpos':'',
    'adim':5,
    'sdim':12,
    'xpos0': np.array([0., 0., 0.05, 0., 0., 0.]), #initialize state dimension to 5 zeros
    'dt': 0.05,
    'substeps': 200,  #6
    'T': 15,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    'additional_viewer': False,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 10,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'init_arm_near_obj': 0.3, #randomize x, y
    'poscontroller_offset':'',
    'posmode':'abs',
    'ztarget':0.13,
    'min_z_lift':0.05,
    'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record',
    'targetpos_clip':[[-0.5, -0.5, -0.08, -np.pi*2, 0], [0.5, 0.5, 0.15, np.pi*2, 0.1]],
    'mode_rel':np.array([True, True, True, True, False]),
    'not_create_goals':''
}

policy = {
    'type' : CluteredGraspPolicy,
    'xyz_std': 3e-1,   #std dev. in xy
    'angle_window': 0.5,   #angle delta chosen uniformly from these values
    'z_window': 0.03,
    'max_norm':0.15
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
