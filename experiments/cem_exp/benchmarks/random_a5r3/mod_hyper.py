from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7
}

