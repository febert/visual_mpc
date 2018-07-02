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
jsondir = base_dir + '/pretrained_models/pushing_weiss'
configuration = {
'pred_model': Alex_Interface_Model,
'setup_predictor':setup_predictor,
'json_dir':jsondir,
'pretrained_model':jsondir + '/modeldata/model-230000',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'batch_size': 200,           #'batch size for training' ,
'adim':5,
'sdim':4,
'ndesig':2,
'orig_size':[56,64],
'img_height':56,
'img_width':64,
'data_dir':'/pushing_data/weiss_gripper_20k/conf.py',
'ncam':1,
}