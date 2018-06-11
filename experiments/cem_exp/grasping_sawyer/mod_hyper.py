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
    'T': 30,
    'adim':4,
    'sdim':5,
    'make_final_gif':'',
    'image_height':48,
    'image_width':64,
    'additional_viewer':'',
    'data_save_dir':current_dir + '/data/train',
    'posmode':"",
    'targetpos_clip':[[-0.5, -0.5, -0.08, -2 * np.pi, -1], [0.5, 0.5, 0.15, 2 * np.pi, 1]],
    'mode_rel':np.array([True, True, True, True, False]),
    'autograsp' : {'reopen':'', 'zthresh':??,'touchthresh':??},
    'cameras':['maincam', 'leftcam'],
    'verbose':"",
    # 'compare_mj_planner_actions':'',
}

policy = {
    'verbose':'',
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 0.02,   #std dev. in xy          #TODO: check these
    'initial_std_lift': 1.6,   #std dev. in xy
    'initial_std_rot' : np.pi / 18,
    'initial_std_grasp' : 0,
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':10,
}

