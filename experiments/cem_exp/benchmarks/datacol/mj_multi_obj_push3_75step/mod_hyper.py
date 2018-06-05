import os
import python_visual_mpc

current_dir = '/'.join(str.split(os.path.abspath(__file__), '/')[:-1])
bench_dir = '/'.join(str.split(os.path.abspath(__file__), '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller import CEM_controller

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo
import numpy as np
folder_name = '/'.join(str.split(__file__, '/')[-2:-1])

agent = {
    'type': AgentMuJoCo,
    'T': 75,
    'substeps':50,
    # 'make_final_gif':'',
    'adim':3,
    'sdim':6,
    'filename': ROOT_DIR + '/mjc_models/cartgripper_updown_whitefingers.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper_updown_whitefingers.xml',
    'gen_xml':1,   #generate xml every nth trajecotry
    'num_objects': 3,
    'object_max_len':0.15,
    'object_min_len':0.08,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_height':48,
    'image_width':64,
    'additional_viewer':'',
    'data_save_dir': current_dir + '/data/train',
    'posmode':"",
    'targetpos_clip':[[-0.45, -0.45, -0.08], [0.45, 0.45, 0.15]],
    'mode_rel':np.array([True, True, True]),
    'not_use_images':"",
    'sample_objectpos':'',
    'const_dist':0.2,
    'randomize_initial_pos':'',
    'first_last_noarm':'',
    'object_mass':0.1,
    'master_datadir':'/raid/ngc2/pushing_data/cartgripper/' + folder_name + '/train',
    'not_write_scores':''
}

policy = {
    # 'verbose':'',
    'type' : CEM_controller,
    'current_dir':current_dir,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 0.08,        # std dev. in xy
    'initial_std_lift': 0.1,
    'iterations': 2,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':10,
    'no_action_bound':"",
    'num_samples': 100,
    'replan_interval':10,
}

config = {
    'current_dir':current_dir,
    'save_data': True,
    'start_index':0,
    'end_index': 200000,
    'traj_per_file':5,
    'agent':agent,
    'policy':policy,
}
