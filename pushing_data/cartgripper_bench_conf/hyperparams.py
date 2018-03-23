""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

import os

current_dir = '/'.join(str.split(__file__, '/')[:-1])

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
ROOT_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': ROOT_DIR + '/mjc_models/cartgripper.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper.xml',
    'data_collection': True,
    'sample_objectpos':'',
    'adim':3,
    'sdim':6,
    'xpos0': np.array([0.0, -0.0, 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'T': 15,
    'skip_first': 40,   #skip first N time steps to let the scene settle
    # 'additional_viewer': 10,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_height' : 48,
    'image_width' : 64,
    'image_channels' : 3,
    'num_objects': 4,
    'novideo':'',
    'gen_xml':1,   #generate xml every nth trajecotry
    'randomize_ballinitpos':'',
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeats': 3, # number of repeats for each action
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 1e-5,   #std dev. in xy
    # 'initial_std_grasp': 1e-5,   #std dev. in xy
    'use_goal_image':''
}

tag_images = {'name': 'images',
             'file':'/images/im{}.png',   # only tindex
             'shape':[agent['image_height'],agent['image_width'],3],
               }

tag_qpos = {'name': 'qpos',
             'shape':[3],
             'file':'/state_action.pkl'}
tag_object_full_pose = {'name': 'object_full_pose',
                         'shape':[4,7],
                         'file':'/state_action.pkl'}
tag_object_statprop = {'name': 'obj_statprop',
                     'not_per_timestep':''}

source_tag_actions = {'name':'actions',
                      'shape': [3],
                      'file':'/state_action.pkl'
                      }

config = {
    'save_raw_images':'',
    'save_data': True,
    'start_index': 0,
    'end_index': 1000,
    'agent': agent,
    'policy': policy,
    'ngroup': 100,
    'sourcetags':[tag_images, tag_qpos, source_tag_actions],
    'sequence_length':agent['T'],
    'source_basedirs':[current_dir + '/train'],
    'current_dir':current_dir,
}
