import os
import python_visual_mpc
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 15,
    'repeat': 1,
    'initial_std': .035,   #std dev. in xy
    'initial_std_lift': 0.,
    'gdnconf': current_dir + '/gdnconf.py',
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':30,
    'goal_image':''
}

agent = {
    'T': 30,
    'adim':3,
    'sdim':6,
    'make_final_gif':'',
    'no_instant_gif':"",
    'verbose':'',
    'bench_conf_pertraj': ROOT_DIR + '/pushing_data/cartgripper_bench_conf/train'  #folder where to load configurations and images
    'ngroup',
    'filename': ROOT_DIR + '/mjc_models/cartgripper.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper.xml',
}