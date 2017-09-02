
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller
policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': .035,
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'verbose':'',
    'predictor_propagation': '',   # use the model get the designated pixel for the next step!
    'action_cost_factor': 0,
    'no_pixdistrib_video':'',
    'no_instant_gif':"",
    'finalweight':10,
    'rew_all_steps':"",
    'ndesig':2,
}

agent = {
    'T': 20,
}