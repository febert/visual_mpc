""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.det_grasp_policy import DeterministicGraspPolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import AgentMuJoCo
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy

current_dir = '/'.join(str.split(__file__, '/')[:-1])
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': folder_name + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_updown_rot.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_updown_rot.xml',
    'not_use_images':"",
    'object_mass':0.01,
    'friction':1.5,
    'adim':4,
    'sdim':8,
    'make_final_gif':'',
    'xpos0': np.array([0., 0., 0.1, 0.]), #initialize state dimension to 5 zeros
    'dt': 0.05,
    'substeps': 200,  #6
    'T': 15,
    'skip_first': 2,   #skip first N time steps to let the scene settle
    'additional_viewer': True,
    'image_height' : 48,
    'image_width' : 64,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_channels' : 3,
    'num_objects': 1,
    'novideo':'',
    'gen_xml':1,   #generate xml every nth trajecotry
    'posmode':"",
    'targetpos_clip':[[-0.45, -0.45, -0.08, -np.pi*2], [0.45, 0.45, 0.15, np.pi*2]],
    'discrete_adim':[2],
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeat': 3,               # number of repeats for each action
    'initial_std': 0.08,        # std dev. in xy
    'initial_std_lift': 1.6, #0.1,
    'initial_std_rot': 0.1,
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

config = {
    'current_dir':current_dir,
    'save_data': False,
    'save_raw_images':'',
    'start_index':0,
    'end_index': 99,
    'agent':agent,
    'policy':policy,
    'ngroup': 500,
    'sourcetags':[tag_images, tag_qpos, tag_object_full_pose, tag_object_statprop],
    'source_basedirs':[os.environ['VMPC_DATA_DIR'] + '/cartgripper/cartgripper_startgoal_short/train'],
    'sequence_length':2
}
