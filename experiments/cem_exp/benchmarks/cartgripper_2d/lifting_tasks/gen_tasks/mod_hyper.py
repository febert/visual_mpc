from python_visual_mpc.visual_mpc_core.agent.create_configs_agent import CreateConfigAgent
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
from python_visual_mpc.visual_mpc_core.algorithm.policy import DummyPolicy
import os


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 96,
    'viewer_image_width': 128,
    'cube_objects': True
}


agent = {
    'type': CreateConfigAgent,
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'image_height': 48,
    'image_width': 64,
    'gen_xml': 1,  # generate xml every nth trajecotry
}


config = {
    'agent': agent,
    'policy': {'type': DummyPolicy},
    'save_raw_images': True,
    'start_index': 0,
    'end_index': 150,
    'ngroup': 1000
}