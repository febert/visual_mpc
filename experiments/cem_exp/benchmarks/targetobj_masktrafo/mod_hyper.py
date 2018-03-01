import os
import python_visual_mpc
current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

from python_visual_mpc.visual_mpc_core.agent.agent_mjc import AgentMuJoCo

agent = {
    'type': AgentMuJoCo,
    'T': 30,
    'substeps':20,
    'adim':3,
    'sdim':6,
    'make_final_gif':'',
    # 'no_instant_gif':"",
    'filename': ROOT_DIR + '/mjc_models/cartgripper.xml',
    'filename_nomarkers': ROOT_DIR + '/mjc_models/cartgripper.xml',
    'gen_xml':1,   #generate xml every nth trajecotry
    'num_objects': 1,
    'viewer_image_height' : 480,
    'viewer_image_width' : 640,
    'image_height':48,
    'image_width':64,
    'additional_viewer':'',
    'data_save_dir':current_dir + '/data/train',
    'get_curr_mask':'',
    'goal_mask':''
}

policy = {
    'verbose':'',
    'type' : CEM_controller,
    'low_level_ctrl': None,
    'current_dir':current_dir,
    'usenet': True,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 10.,   #std dev. in xy
    'initial_std_lift': 1e-5,   #std dev. in xy
    'gdnconf': current_dir + '/gdnconf.py',
    'netconf': current_dir + '/conf.py',
    'iterations': 3,
    'action_cost_factor': 0,
    'rew_all_steps':"",
    'finalweight':10,
    'no_action_bound':"",
    'predictor_propagation': '',   # use the model get the designated pixel for the next step!
    'masktrafo_obj':'',            # transform ground truth target object mask
}

tag_images = {'name': 'images',
             'file':'/images/im{}.png',   # only tindex
             'shape':[agent['image_height'],agent['image_width'],3],
               }
tag_qpos = {'name': 'qpos',
             'shape':[3],
             'file':'/state_action.pkl'}
tag_object_full_pose = {'name': 'object_full_pose',
                         'shape':[4,7],
                         'file':'/state_action.pkl'}
tag_object_statprop = {'name': 'obj_statprop',
                     'not_per_timestep':''}
goal_mask = {'name': 'goal_mask',
             'not_per_timestep':''}

config = {
    'save_data': False,
    'save_raw_images':'',
    'start_index':0,
    'end_index': 50, #1000,
    'agent':agent,
    'policy':policy,
    'ngroup': 100,
    'sourcetags':[tag_images, tag_qpos, tag_object_full_pose, tag_object_statprop, goal_mask],
    'source_basedirs':[ROOT_DIR + '/pushing_data/cartgripper_startgoal_masks/train'],
    'sequence_length':2
}