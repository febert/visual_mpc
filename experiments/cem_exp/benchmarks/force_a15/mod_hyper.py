from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 15,
    'repeat': 1,
    'initial_std': 7,
    'num_samples': 200
}

