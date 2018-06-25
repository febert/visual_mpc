
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller
import numpy as np


agent = {
    'sawyer':'',
    'T': 15,
    'adim':5,
    'sdim':4,
    'image_height':56,
    'image_width':64,
    'ndesig':1,
    'make_final_gif':'',
    'make_final_vid':'',
    'wristrot':'',
    'startpos_basedon_click':'',
    'record':current_dir + '/verbose',
    'discrete_adim':[2],
    'save_pkl':''
}


policy = {
    'verbose':'',
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,   #std dev. in xy
    'initial_std_grasp': 0.,
    'initial_std_lift': 1.,
    'initial_std_rot': 0.,
    'netconf': current_dir + '/conf.py',
    'gdnconf': current_dir + '/gdnconf.py',
    'iterations': 3,
    'action_cost_factor': 0,
    'no_instant_gif':"",
    'rew_all_steps':"",
    'finalweight':30,
    'no_pixdistrib_video':'',
    'use_goal_image':'',
    'register_gtruth':['start','goal'],
    'trade_off_reg':'',
    ### start reuse action
    'replan_interval':3,
    'reuse_mean':'',
    'reuse_action_as_mean':"",
    'reduce_std_dev':0.5, #0.2 # reduce standard dev in later timesteps when reusing action
    'num_samples': [400,200],
    'selection_frac':0.05,
    ### end resuse action
}

