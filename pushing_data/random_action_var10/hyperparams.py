""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np


from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

num_objects = 4

SENSOR_DIMS = {
    "JOINT_ANGLES": 2+ 7*num_objects,  #adding 7 dof for position and orientation for every free object
    "JOINT_VELOCITIES": 2+ 6*num_objects,  #adding 6 dof for speed and angular vel for every free object; 2 + 6 = 8
    "END_EFFECTOR_POINTS": 3,
    "END_EFFECTOR_POINT_VELOCITIES": 3,
    "ACTION": 2,
    # RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    "RGB_IMAGE_SIZE": 3,
}

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
VISUAL_MPC_DIR = '/'.join(str.split(__file__, '/')[:-3])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': BASE_DIR,
    'data_files_dir': BASE_DIR + '/train',
    'target_filename': BASE_DIR + 'target.npz',
    'log_filename': BASE_DIR + 'log.txt',
    'conditions': 1,
    'no_sample_logging': True,
}

if not os.path.exists(common['data_files_dir']):
    raise ValueError('data files directory not found!')

agent = {
    'type': AgentMuJoCo,
    'filename': VISUAL_MPC_DIR +'/mjc_models/pushing2d.xml',
    'filename_nomarkers': VISUAL_MPC_DIR +'/mjc_models/pushing2d.xml',
    'data_collection': True,
    'x0': np.array([0., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 20,  #6
    'conditions': common['conditions'],
    'T': 15,
    'skip_first': 5,   #skip first N time steps to let the scene settle

    #adding 7 dof for position and orentation of free object
    'additional_viewer': True,
    'image_dir': common['data_files_dir'] + "imagedata_file",
    'image_height' : IMAGE_HEIGHT,
    'image_width' : IMAGE_WIDTH,
    'image_channels' : IMAGE_CHANNELS,
    'num_objects': num_objects,
    'novideo':''
}

policy = {
    'type' : Randompolicy,
    'initial_var': 10,
    'numactions': 5, # number of consecutive actions
    'repeats': 3, # number of repeats for each action
}

config = {
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'policy': policy
}

# common['info'] = generate_experiment_info(config)
