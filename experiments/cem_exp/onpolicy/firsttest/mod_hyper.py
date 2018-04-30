import os
import python_visual_mpc
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
import numpy as np
agent = {
    'type': AgentMuJoCo,
    'T': 6, #############################30,
    'substeps':200,
    'adim':3,
    'sdim':6,
    'make_final_gif':'',
    # 'no_instant_gif':"",
    'filename': ROOT_DIR + '/mjc_models/cartgripper_updown.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper_updown.xml',
    'gen_xml':1,   #generate xml every nth trajecotry
    'skip_first':10,
    'num_objects': 1,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_height':48,
    'image_width':64,
    'sample_objectpos':'',
    'randomize_ballinitpos':'',
    'const_dist':0.2,
    'data_save_dir':current_dir + '/data/train',
    'posmode':"",
    'targetpos_clip':[[-0.45, -0.45, -0.08], [0.45, 0.45, 0.15]],
    'discrete_adim':[2],
}

policy = {
    # 'verbose':'',
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 2,############################5,
    'repeat': 3,
    'initial_std': 0.08,        # std dev. in xy
    'initial_std_lift': 2.5,
    'netconf': current_dir + '/conf.py',
    'iterations': 1,######################################3,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':10,
    # 'predictor_propagation': '',   # use the model get the designated pixel for the next step!
}

onpolconf = {
    'infnet_reload_freq':10, #############,     # reload inference model weights after n number of new trajectores collected
    'replay_size':2,
    'prefil_replay':1,        # fill replay with existing trajectories from dataset
}

config = {
    'current_dir':current_dir,
    'save_data': True,
    'start_index':0,
    'end_index': 59999,
    'agent':agent,
    'policy':policy,
    'onpolconf':onpolconf,
}