
current_dir = '/'.join(str.split(__file__, '/')[:-1])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'verbose':'',
    'action_cost_factor': 0,
    'no_instant_gif':"",
    'rew_all_steps':"",
    'finalweight':10,
    'no_pixdistrib_video':'',
    'ndesig':2
}

agent = {
    'T': 20,
    'adim':5,
    'sdim':4,
    'make_final_gif':'',
    'wristrot':'',
    'opencv_tracking':'',
}