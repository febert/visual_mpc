
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'load_goal_image':'make_easy_goal',
    'verbose':'',
    # 'predictor_propagation': '',   # use the model get the designated pixel for the next step!
    'action_cost_factor': 0,
    # 'reuse_mean_cov':'',
    # 'no_instant_gif':"",
    # 'rew_all_steps':"",
    # 'finalweight':10,
    'use_first_plan':'',
    'avoid_occlusions':20,
}

agent = {
    'T': 15,
}