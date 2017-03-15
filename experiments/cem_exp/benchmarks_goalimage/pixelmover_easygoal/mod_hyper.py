

current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py',
    'iterations': 5,
    'use_first_plan': False, # execute MPC instead using firs plan
    'load_goal_image':'make_easy_goal'
}

agent = {
    'T': 25,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal'  #select file with different starting configurations
}