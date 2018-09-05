from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
import os


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128
}

agent = {
    'type': GeneralAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height': 48,
    'image_width': 64,
    'gen_xml': 1,  # generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'rejection_sample': 1,
    'make_final_gif': True
}

policy = {
    'type': Randompolicy,
    'nactions': 10,
    'action_order': ['x', 'z', 'grasp'],
    'initial_std_lift': 0.5,  # std dev. in xy
}

config = {
    'traj_per_file': 16,
    'seperate_good': True,
    'current_dir': current_dir,
    'save_raw_images': True,
    'save_data': True,
    'start_index':0,
    'end_index': 60000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}