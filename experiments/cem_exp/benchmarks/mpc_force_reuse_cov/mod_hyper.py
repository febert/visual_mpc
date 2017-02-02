

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
    'iterations': 10,
    # 'exp_factor': 10,   #last trial 1.2    # activates mean and covariance reuse, comment out to deactivate
    'num_samples': 200,
    # 'reduce_iter': 5
}


agent = {
    'T': 25
}
