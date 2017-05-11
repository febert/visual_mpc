import os
current_dir = os.path.dirname(os.path.realpath(__file__))
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'currentdir':str(current_dir),
    'use_goalimage':"",
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'use_first_plan': False, # execute MPC instead using firs plan
    'iterations': 1, ####################
    'load_goal_image':'make_easy_goal',
    'rewardnetconf':current_dir + '/rewardconf.py',   #configuration for reward network
    'rewardmodel_sequence_length':15,
    'num_samples': 200,
    'mujoco_with_rewardnet':'',
    # 'verbose':''
}

agent = {
    'T': 25,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal',
    # 'novideo': ''
}