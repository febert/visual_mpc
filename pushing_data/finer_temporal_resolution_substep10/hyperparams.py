""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from lsdc import __file__ as gps_filepath
from lsdc.agent.mjc.agent_mjc import AgentMuJoCo
from lsdc.algorithm.policy.random_impedance_point import Random_impedance_point
from lsdc.algorithm.policy.random_policy import Randompolicy

from lsdc.gui.config import generate_experiment_info

from lsdc.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

num_objects = 4

SENSOR_DIMS = {
    JOINT_ANGLES: 2+ 7*num_objects,  #adding 7 dof for position and orientation for every free object
    JOINT_VELOCITIES: 2+ 6*num_objects,  #adding 6 dof for speed and angular vel for every free object; 2 + 6 = 8
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
    # RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/lsdc_exp/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': '/home/frederik/Documents/pushing_data/finer_temporal_resolution_substep10/train/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'no_sample_logging': True,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pushing2d.xml',
    'filename_nomarkers': './mjc_models/pushing2d.xml',
    'data_collection': True,
    'x0': np.array([0., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 10,  #6
    'numactions': 5, # number of consecutive actions
    'repeat': 6, # number of repeats for each action
    'conditions': common['conditions'],
    'T': 30,
    'skip_first': 5,   #skip first N time steps to let the scene settle
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],

    #adding 7 dof for position and orentation of free object
    'joint_angles': SENSOR_DIMS[JOINT_ANGLES],
    'joint_velocities': SENSOR_DIMS[JOINT_VELOCITIES],
    'additional_viewer': True,
    'image_dir': common['data_files_dir'] + "imagedata_file",
    'image_height' : IMAGE_HEIGHT,
    'image_width' : IMAGE_WIDTH,
    'image_channels' : IMAGE_CHANNELS,
    'num_objects': num_objects,
    'record':False
}

policy = {
    # 'type' : Random_impedance_point
    'type' : Randompolicy,
    'initial_var': 40
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
