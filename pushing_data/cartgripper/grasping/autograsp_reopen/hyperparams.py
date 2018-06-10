""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path

import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.random_policy import RandomPickPolicy
from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
from python_visual_mpc.visual_mpc_core.infrastructure.utility.tfrecord_from_file import DefaultTraj
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
DATA_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])

def convert_to_record(state_action):
    loaded_traj = DefaultTraj()

    loaded_traj.actions = state_action['actions']
    touch_sensors = state_action['finger_sensors']
    loaded_traj.X_Xdot_full = np.concatenate((state_action['target_qpos'][:-1, :], touch_sensors), axis = 1)

    good_lift = False

    valid_frames = np.logical_and(state_action['target_qpos'][1:, -1] > 0, np.logical_and(touch_sensors[:, 0] > 0, touch_sensors[:, 1] > 0))
    off_ground = state_action['target_qpos'][1:,2] >= 0
    object_poses = state_action['object_full_pose']

    if any(np.logical_and(valid_frames, off_ground)):
        obj_eq = object_poses[0, :, :2] == state_action['obj_start_end_pos']
        obj_eq = np.logical_and(obj_eq[:, 0], obj_eq[:, 1])
        obj_eq = np.argmax(obj_eq)
        obj_max =  np.amax(object_poses[:,obj_eq,2])
        if obj_max >=0:
            good_lift = True

    return good_lift, loaded_traj

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
    'cameras':['maincam', 'leftcam'],
    'finger_sensors' : True,
    'randomize_initial_pos':'',
    'arm_start_lifted':0.15,
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
    'num_objects': 4,
    'novideo':'',
    'gen_xml':10,   #generate xml every nth trajecotry
    'pos_disp_range': 0.5, #randomize x, y
    'poscontroller_offset':'',
    'posmode':'abs',
    'ztarget':0.13,
    'min_z_lift':0.05,
    'make_final_gif':'', #keep this key in if you want final gif to be created
    'record': BASE_DIR + '/record/',
    'targetpos_clip':[[-0.5, -0.5, -0.08, -2 * np.pi, -1], [0.5, 0.5, 0.15, 2 * np.pi, 1]],
    'mode_rel':np.array([True, True, True, True, False]),
    'discrete_gripper' : -1, #discretized gripper dimension,
    'lift_rejection_sample' : 15,
    'object_mass' : 0.1,
    'friction' : 1.0,
    'autograsp' : {'reopen':'', 'zthresh':0.087,'touchthresh':0.0},
    'reopen':'',
    'master_datadir' : '/raid/ngc2/grasping_data/autograsp_reopen/'
#    'file_to_record' : convert_to_record
    #'object_meshes':['giraffe'] #folder to original object + convex approximation
}

policy = {
    'type' : RandomPickPolicy,
    'nactions' : 5,
    'repeat' : 3,
    'no_action_bound' : False, 
    'initial_std': 0.02,   #std dev. in xy
    'initial_std_lift': 1.6,   #std dev. in xy
    'initial_std_rot' : np.pi / 18,
    'initial_std_grasp' : 2 
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
