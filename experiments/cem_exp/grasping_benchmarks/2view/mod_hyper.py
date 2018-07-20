import os
import python_visual_mpc
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

from python_visual_mpc.visual_mpc_core.agent.general_agent import AgentMuJoCo
import numpy as np
agent = {
    'type': AgentMuJoCo,
    'T': 5,  #####################
    'substeps':200,
    'adim':5,
    'sdim':12,
    'make_final_gif':'',
    'filename': ROOT_DIR + '/mjc_models/cartgripper_grasp.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper_grasp.xml',
    'gen_xml':1,   #generate xml every nth trajecotry
    'num_objects': 1,
    'object_mass':0.01,
    'friction':1.5,
    'skip_first':0,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_height':48,
    'image_width':64,
    'additional_viewer':'',
    'data_save_dir':current_dir + '/data/train',
    'posmode':"",
    # 'targetpos_clip':[[-0.45, -0.45, -0.08, -np.pi*2, -100], [0.45, 0.45, 0.15, np.pi*2, 100]], ###########3
    'targetpos_clip':[[-0.45, -0.45, -0.08, -np.pi*2, 0.], [0.45, 0.45, 0.15, np.pi*2, 0.1]],
    'mode_rel':np.array([True, True, True, True, False]),
    'cameras':['maincam', 'leftcam'],
    'verbose':"",
    # 'compare_mj_planner_actions':'',
}

policy = {
    'verbose':'',
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 0.05,        # std dev. in xy
    'initial_std_lift': 0.01,
    'initial_std_rot': 0.01,
    'initial_std_grasp': 30,  #######
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':10,
    # 'no_action_bound':"",
}

tag_images0 = {'name': 'images0',
             'file':'/images0/im{}.png',   # only tindex
             'shape':[agent['image_height'],agent['image_width'],3],
               }

tag_images1 = {'name': 'images1',
              'file':'/images1/im{}.png',   # only tindex
              'shape':[agent['image_height'],agent['image_width'],3],
              }

tag_qpos = {'name': 'qpos',
             'shape':[6],
             'file':'/state_action.pkl'}

tag_actions = {'name': 'actions',
            'shape':[15,5],
            'not_per_timestep':'',
            'file':'/state_action.pkl'}

tag_object_full_pose = {'name': 'object_full_pose',
                         'shape':[4,7],
                         'file':'/state_action.pkl'}
tag_object_statprop = {'name': 'obj_statprop',
                     'not_per_timestep':''}

config = {
    'current_dir':current_dir,
    'save_data': False,
    'save_raw_images':'',
    'start_index':0,
    'end_index': 49,
    'agent':agent,
    'policy':policy,
    'ngroup': 100,
    'sourcetags':[tag_images0, tag_images1, tag_qpos, tag_object_full_pose, tag_object_statprop, tag_actions],
    'source_basedirs':[os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/cartgripper_startgoal_2view_lift_above_obj/train'],
    'sequence_length':2
}