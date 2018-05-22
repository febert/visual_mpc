
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])


from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
import numpy as np

agent = {
    'T': 30,
    'adim':5,
    'sdim':4,
    'ndesig':1,
    'wristrot':'',
}

policy = {
    'type' : Randompolicy,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,   #std dev. in xy
    'initial_std_grasp': 0.1,
    'initial_std_lift': 0.1,
    'initial_std_rot': 0.1,
}

