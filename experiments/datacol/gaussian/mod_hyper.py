
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])


from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
import numpy as np

agent = {
    'T': 15,
    'adim':5,
    'sdim':4,
    'ndesig':1,
    'wristrot':'',
    'collect_data':''
}


policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,   #std dev. in xy
    'initial_std_lift': 0.1,
    'initial_std_rot': 0.3,
    'initial_std_grasp': 0.1,
}


dataconf = {
'sourcetags': ['/main'],               # list of source dirs for different camera view-points
'total_num_img': 96,
'take_ev_nth_step':2,                          # subsample trajectories
'target_res': (64,64),                             #128x128
'shrink_before_crop':1/9.,
'rowstart':10,
'colstart':32,
'img_height':48,
'img_width':64,
'adim':agent['adim'],
'sdim':agent['sdim'],
}
