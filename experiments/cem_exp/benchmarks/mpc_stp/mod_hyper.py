

current_dir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'iterations' : 5,
    'netconf': current_dir + '/conf.py',
    'use_first_plan': False, # execute MPC instead using firs plan
}

agent = {
    'T': 25   # important for MPC
}