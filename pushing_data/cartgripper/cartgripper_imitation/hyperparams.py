""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division
import os.path

import numpy as np

from python_visual_mpc.imitation_model.imitation_policy import ImitationPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
current_dir = os.path.dirname(os.path.realpath(__file__))

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'not_use_images':"",
    'visible_viewer':True,
    'sample_objectpos':'',
    'adim':5,
    'sdim':12,
    'xpos0': np.array([0., 0., 0.05, 0., 0., 0.]), #initialize state dimension to 5 zeros
    'dt': 0.05,
    'substeps': 200,  #6
    'T': 20,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    'additional_viewer': False,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'randomize_initial_pos':'', #randomize x, y
    'poscontroller_offset':'',
    'posmode':'abs',
    'ztarget':0.13,
    'drop_thresh':0.02,
    #'make_final_gif':True,
    'record': BASE_DIR + '/record/',
    'targetpos_clip':[[-0.5, -0.5, -0.08, -np.pi*2, 0], [0.5, 0.5, 0.15, np.pi*2, 0.1]],
    'mode_rel':np.array([True, True, True, True, False]),
    #'object_meshes':['hubble_model_kit_1'] #folder to original object + convex approximation
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : ImitationPolicy,
    'net_config' : os.path.join(BASE_DIR, 'conf_lstm_mdn_states.py'),
    'pretrained' :'modelfinal', #'model65000',
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 80000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
