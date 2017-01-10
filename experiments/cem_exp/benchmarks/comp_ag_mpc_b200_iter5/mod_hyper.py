

currentdir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'netconf': currentdir + '/conf.py',
    'iterations': 5,
    'num_samples': 200, #M
    'use_first_plan': False # execute MPC instead using firs plan
}

agent = {
    'T': 25   # important for MPC
}