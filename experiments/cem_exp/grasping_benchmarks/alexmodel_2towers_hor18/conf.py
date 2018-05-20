import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
# tf record data location:

# local output directory
OUT_DIR = current_dir + '/modeldata'

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model
import video_prediction
base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor
jsondir = base_dir + '/pretrained_models/bair_action_free/grasp_push_2views'

sequence_length = 18
override_json = {
    'sequence_length':sequence_length
}

configuration = {
'override_json':override_json,
'pred_model': Alex_Interface_Model,
'setup_predictor':setup_predictor,
'json_dir':jsondir,
'pretrained_model':jsondir + '/model-190000',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': sequence_length,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 2000,
'adim':5,
'sdim':12,
'orig_size':[48,64],
'img_height':48,
'img_width':64,
'ndesig':1,
'use_vel':'',                # add the velocity to the state input
'ncam':2,
}