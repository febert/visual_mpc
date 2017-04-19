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
    'num_samples': 50,
    'iterations': 3,
    # 'parallel_smp':'',
    'n_reseed':1
}

agent = {
    'T': 25,
    'save_goal_image': "",
    'current_dir': current_dir,
    'start_confs': current_dir + '/configs_easy_goal_1e4'
}

config = {
    'save_data': True,
    'traj_per_file': 20
}
common = {
    'data_files_dir': current_dir + '/tfrecords/train'
}
