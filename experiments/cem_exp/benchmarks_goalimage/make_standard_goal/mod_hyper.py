

current_dir = '/'.join(str.split(__file__, '/')[:-1])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'use_first_plan': False, # execute MPC instead using firs plan
    'num_samples': 200,
    'iterations': 5
}


agent = {
    'T': 25,
    'save_goal_image': "",
    'current_dir': current_dir,
}