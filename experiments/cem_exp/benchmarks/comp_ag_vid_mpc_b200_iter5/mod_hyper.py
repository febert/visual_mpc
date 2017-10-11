
current_dir = '/'.join(str.split(__file__, '/')[:-1])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller import CEM_controller


policy = {
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'usenet':'',
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'iterations' : 5,
    'netconf': current_dir + '/conf.py',
    # 'use_first_plan':  # execute MPC instead using firs plan
}

agent = {
    'T': 2 ######25   # important for MPC
}