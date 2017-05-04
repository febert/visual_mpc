current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'use_goalimage':"",
    'low_level_ctrl': None,
    'usenet': False,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'use_first_plan': False, # execute MPC instead using firs plan
    'iterations': 5,
    'load_goal_image':'make_easy_goal',
    'rewardnetconf':current_dir + '/rewardconf.py',   #configuration for reward network
    'rewardmodel_sequence_length':25,
    'num_samples': 50,
    'mujoco_with_rewardnet':''
}

agent = {
    'T': 25,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal',
    'novideo':''
}