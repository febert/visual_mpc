

current_dir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'goalimage':"",
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 2,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py',
    'use_first_plan': False # execute MPC instead using firs plan
}

agent = {
    'T': 25,
}