

current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'use_goalimage':"",
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py',
    'use_first_plan': False, # execute MPC instead using firs plan
    'iterations': 5,
    'load_goal_image':'make_easy_goal',
    'usepixelerror':''
}

agent = {
    'T': 25,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal',
    'store_video_prediction': 10 # the number of predicted terminal frames stored, equal to K selected trajectories in CEM
}

config = {
    'save_data': True,
    'traj_per_file': 20
}
common = {
    'data_files_dir': current_dir + '/tfrecords'
}