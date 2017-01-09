
currentdir = '/'.join(str.split(__file__, '/')[:-1])


from lsdc.algorithm.policy.pos_controller import Pos_Controller
low_level_conf = {
    'type': Pos_Controller,
    'mode': 'relative',
    'randomtargets' : False
}


from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': low_level_conf,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 0.15,
    'netconf': currentdir + '/conf.py',
    'use_first_plan': True # execute MPC instead using firs plan
}