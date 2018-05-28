import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper/grasping/dualcam_pick_place_dataset/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

from python_visual_mpc.video_prediction.dynamic_rnn_model.multi_view_model import Multi_View_Model
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor

import python_visual_mpc
base_dir = python_visual_mpc.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])


configuration = {
'experiment_name': 'rndaction_var10',
'pred_model':Multi_View_Model,
'pretrained_model':base_dir + '/tensorflow_data/sim/multi_view_models/multi_view_grasp_push/modeldata/model132000',
'setup_predictor':setup_predictor,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir, #'directory for writing summary.' ,
'num_iterations': 200000,   #'number of training iterations.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_transformed_images': 1,   # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,      # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 2000,           #'batch size for training' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'sawyer':'',
'single_view':"",
'1stimg_bckgd':'',
'visual_flowvec':'',
'adim':5,
'sdim':12,
'normalization':'in',
'previmg_bckgd':'',
'orig_size':[48,64],
'ncam':2,
'ndesig':1,
'use_vel':'',                # add the velocity to the state input
}
