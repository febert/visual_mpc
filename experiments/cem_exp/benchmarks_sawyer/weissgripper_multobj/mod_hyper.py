
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller
import numpy as np

policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,   #std dev. in xy
    'initial_std_grasp': 0.,   #std dev. in xy
    'initial_std_lift': 1.,   #std dev. in xy
    'initial_std_rot': 0.,   #std dev. in xy
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'verbose':'',
    'predictor_propagation': '',   # use the model get the designated pixel for the next step!
    'action_cost_factor': 0,
    'no_instant_gif':"",
    'rew_all_steps':"",
    'finalweight':10,
    'no_pixdistrib_video':'',
}

agent = {
    'ndesig':2,
    # 'opencv_tracking':'',
    'T': 20,
    'adim':5,
    'sdim':4,
    'state_dim':4,
    'make_final_gif':'',
    'wristrot':''
}