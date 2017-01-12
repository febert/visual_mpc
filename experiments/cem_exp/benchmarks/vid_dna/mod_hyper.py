

current_dir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 15,
    'repeat': 1,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py'
}

