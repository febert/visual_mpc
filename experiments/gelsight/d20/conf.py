
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
# tf record data location:

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model
import video_prediction
base_dir = video_prediction.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor
jsondir = base_dir + '/logs/dice_model_2_14/model.savp.None'

override_json = {'sequence_length':18}

configuration = {
'pred_model': Alex_Interface_Model,
'setup_predictor':setup_predictor,
'override_json':override_json,
'json_dir': jsondir,
'pretrained_model':jsondir + '/model-150000',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': override_json['sequence_length'],      # 'sequence length to load, including context frames.' ,
'context_frames': 3,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 200,           #TODO 'batch size for training' ,
'adim':3,
'sdim':3,
'orig_size':[48,64],
'ndesig':1,
#'use_vel':'',                # add the velocity to the state input
'ncam':1,
}
