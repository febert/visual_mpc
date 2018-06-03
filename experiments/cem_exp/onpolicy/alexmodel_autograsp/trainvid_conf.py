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

DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper/onpolicy/alexmodel_autograsp'
PRELOAD_DATA_DIR = {os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/dctouch_openloop_autograsp/good' : 0.5,
                    os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/dctouch_openloop_autograsp/bad' : 0.5}

onpolconf = {
    'save_interval':200,
    'replay_size':{'train':40000, 'val':4000},
    'fill_replay_fromsaved':{'train':20, 'val':20},##################000,              # fill replay with existing trajectories from dataset
}

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model
import video_prediction
base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor
jsondir = base_dir + '/pretrained_models/autograsp'

config = {
'current_dir':current_dir,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'preload_data_dir':PRELOAD_DATA_DIR,
'data_dir':DATA_DIR,
'skip_frame':1,
'onpolconf':onpolconf,
'pred_model': Alex_Interface_Model,
'setup_predictor':setup_predictor,
'json_dir':jsondir,
'pretrained_model':jsondir + '/model.multi_savp.None/model-90000',     # 'filepath of a pretrained model to resume training from.' ,
'ndesig':1,
'orig_size':[48,64],
'sequence_length':15,
'batch_size':16,
'sdim':7,
'adim':5,
'autograsp':'',
'ncam':2,
'num_iterations':200000,
}