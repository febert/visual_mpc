from lsdc.algorithm.policy.random_policy import Randompolicy


current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

policy = {
    'type' : Randompolicy,
    'initial_var': 10,
    'numactions': 8, # number of consecutive actions
    'repeats': 3, # number of repeats for each action
    'load_goal_image':'make_easy_goal',
    'use_goalimage': "",
}

agent = {
    'T': 24,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal',
    'random_baseline': True
}