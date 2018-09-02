from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
from python_visual_mpc.visual_mpc_core.algorithm.random_policy import Randompolicy
from python_visual_mpc.visual_mpc_core.agent.general_agent import GeneralAgent
import os


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {'clean_xml': False}

agent = {
    'type': GeneralAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 15,
    'image_height': 48,
    'image_width': 64,
    'gen_xml': 400,  # generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'make_final_gif': '',
}

policy = {
    'type': Randompolicy,
}

config = {
    'traj_per_file': 16,
    'seperate_good': True,
    'current_dir': current_dir,
    'save_raw_images': True,
    'save_data': True,
    'start_index':0,
    'end_index': 50,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}