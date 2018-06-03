import os
current_dir = os.path.dirname(os.path.abspath(__file__))

import python_visual_mpc
base_dir = python_visual_mpc.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
# tf record data location:

# local output directory
OUT_DIR = current_dir + '/modeldata'

from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor

DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper/onpolicy/oldmodel_autograsp'
PRELOAD_DATA_DIR = {os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/dctouch_openloop_autograsp/good' : 0.5,
                    os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/dctouch_openloop_autograsp/bad' : 0.5}

onpolconf = {
    'save_interval':200,
    'replay_size':{'train':40000, 'val':4000},
    'fill_replay_fromsaved':{'train':20, 'val':20},##################000,              # fill replay with existing trajectories from dataset
}

config = {
'experiment_name': 'rndaction_var10',
'pred_model':Dynamic_Base_Model,
'data_dir':DATA_DIR,
'preload_data_dir':PRELOAD_DATA_DIR,
'num_iterations':200000,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir, #'directory for writing summary.' ,
'pretrained_model':base_dir + '/tensorflow_data/sim/multi_view_models/autograsp/modeldata/model116000',
'sequence_length': 15, #30 ############# 'sequence length to load, including context frames.' ,
'use_len': 15,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_transformed_images': 1,   # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,      # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 16,           #'batch size for training' ,
'learning_rate': 1e-4,     #'the base learning rate of the generator' ,
'1stimg_bckgd':'',
'adim':5,
'sdim':7,
'autograsp':'',
'no_touch':'',
'normalization':'in',
'previmg_bckgd':'',
'orig_size':[48,64],
'img_height':48,
'img_width':64,
'use_vel':'',                # add the velocity to the state input
'onpolconf':onpolconf,
}