
current_dir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py',
    'use_first_plan': False, # execute MPC instead using firs plan
    'iterations': 2,  #5
    'use_corrector': False,
    'current_dir': current_dir,
    'rec_distrib': '',
    'predictor_propagation': True
}

agent = {
    'T': 3, #25,
    'current_dir': current_dir
}