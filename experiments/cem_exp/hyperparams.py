""" Hyperparameters for Large Scale Data Collection (LSDC) """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from lsdc import __file__ as gps_filepath
from lsdc.agent.mjc.agent_mjc import AgentMuJoCo

# from conf import configuration as netconfig
import imp





from lsdc.gui.config import generate_experiment_info

from lsdc.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

num_objects = 1

SENSOR_DIMS = {
    JOINT_ANGLES: 2+ 7*num_objects +2,  #adding 7 dof for position and orientation for every free object + 3 for goal_geom and reference points
    JOINT_VELOCITIES: 2+ 6*num_objects +2,  #adding 6 dof for speed and angular vel for every free object;
    ACTION: 2,
    # RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}

current_dir = '/'.join(str.split(__file__, '/')[:-1])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': current_dir,
    'data_files_dir': '/tmp/',
    'target_filename': current_dir + 'target.npz',
    'log_filename': current_dir + 'log.txt',
    'conditions': 1,
    'no_sample_logging': True,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

alpha = 30*np.pi/180
agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pushing2d_controller.xml',
    'filename_nomarkers': './mjc_models/pushing2d_controller_nomarkers.xml',
    'data_collection': False,
    'x0': np.array([0., 0., 0., 0.,
                    .1, .1, 0., np.cos(alpha/2), 0, 0, np.sin(alpha/2)  #object pose (x,y,z, quat)
                     ]),
    'dt': 0.05,
    'substeps': 20,  #10
    'conditions': common['conditions'],
    'T': 3,
    'skip_first': 5,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'joint_angles': SENSOR_DIMS[JOINT_ANGLES],  #adding 7 dof for position and orentation of free object
    'joint_velocities': SENSOR_DIMS[JOINT_VELOCITIES],
    'additional_viewer': True,
    'image_dir': common['data_files_dir'] + "imagedata_file",
    'image_height' : IMAGE_HEIGHT,
    'image_width' : IMAGE_WIDTH,
    'image_channels' : IMAGE_CHANNELS,
    'num_objects': num_objects,
    'goal_point': np.array([-0.2, -0.2]),
    'current_dir': current_dir,
    'record': current_dir + '/videos/rec'
}


from lsdc.algorithm.policy.pos_controller import Pos_Controller
low_level_conf = {
    'type': Pos_Controller,
    'mode': 'relative',
    'randomtargets' : False
}

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'netconf': current_dir + '/conf.py',
    # 'correctorconf': current_dir + '/cor_conf.py',  #comment this out to switch off
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'use_first_plan': True,
    'num_samples': 200,
    'iterations': 5,
    # 'predictor_propagation': True,
    'current_dir': current_dir,
    # 'rec_distrib': current_dir + '/videos_distrib/distrib'
}

if policy['low_level_ctrl'] == None:
    policy['initial_std'] = 7
elif policy['low_level_ctrl']['type'] == Pos_Controller:
    policy['initial_std'] = 0.15


config = {
    'save_data': False,
    'start_index':0,
    'end_index': 1,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'policy': policy
}

# common['info'] = generate_experiment_info(config)
