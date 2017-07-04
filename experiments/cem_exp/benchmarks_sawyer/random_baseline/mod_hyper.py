

current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'random_policy':'',
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,
    'iterations': 1,
    'use_first_plan':'',
    'no_pixdistrib_video':''
}

agent = {
    'T': 15,
    # 'use_goalimage':"",
}