

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
    'load_goal_image':'make_standard_goal_1e4',
    'usepixelerror':'',
    'n_reseed':1
}

agent = {
    'T': 25,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_standard_goal_1e4/configs_standard_goal_1e4',
    'store_video_prediction': "" # the number of predicted terminal frames stored, equal to K selected trajectories in CEM
}

config = {
    'save_data': True,
    'traj_per_file': 20
}
common = {
    'data_files_dir': current_dir + '/tfrecords/train'
}