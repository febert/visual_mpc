
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
    # 'use_first_plan':'', # execute MPC instead using firs plan
    'iterations': 3,
    'rew_all_steps': "",
    'finalweight':1,
    # 'verbose':"",
    'action_cost_factor':1e-5
}

agent = {
    'T': 10,
}