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
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])
ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])
MODEL_BASE_DIR = ROOT_DIR + '/pushing_data/cartgripper_imitation/'

agent = {
    'type': AgentMuJoCo,
    'data_save_dir': BASE_DIR + '/train',
    'filename': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': DATA_DIR+'/mjc_models/cartgripper_grasp.xml',
    'not_use_images':"",
    'visible_viewer':True,
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
    'num_objects': 1,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'poscontroller_offset':'',
    'posmode':'abs',
    'ztarget':0.13,
    'drop_thresh':0.02,
    'make_final_gif':True,
    'record': BASE_DIR + '/record/',
    'targetpos_clip':[[-0.5, -0.5, -0.08, -np.pi*2, 0], [0.5, 0.5, 0.15, np.pi*2, 0.1]],
    'mode_rel':np.array([True, True, True, True, False]),
    #'object_meshes':['hubble_model_kit_1'] #folder to original object + convex approximation
    # 'displacement_threshold':0.1,
}

policy = {
    'type' : ImitationPolicy,
    'net_config' : os.path.join(MODEL_BASE_DIR, 'conf_attention_lstm.py'),
    'pretrained' :'model20000', #'model65000',
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
    'end_index': 49, #1000,
    'agent':agent,
    'policy':policy,
    'ngroup': 100,
    'sourcetags':[tag_images, tag_qpos, tag_object_full_pose, tag_object_statprop],
    'source_basedirs':[ROOT_DIR + '/pushing_data/cartgripper_lift_benchmark/train'],
    'sequence_length':2
}
