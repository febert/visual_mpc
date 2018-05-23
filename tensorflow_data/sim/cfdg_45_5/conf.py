import os
current_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np

# tf record data location:
# DATA_DIR = {os.environ['VMPC_DATA_DIR'] + '/cartgripper/train' : 0.5, 
#             os.environ['VMPC_DATA_DIR'] + '/benchmarks/good' : 0.05, 
#             os.environ['VMPC_DATA_DIR'] + '/benchmarks/bad' : 0.45}
DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/benchmarks/good'
# local output directory
OUT_DIR = current_dir + '/modeldata'
IMITATION_BASE_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper_imitation_openloop/'

from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model
from python_visual_mpc.imitation_model.setup_imitation import setup_openloop_predictor
#from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor

def state_conv(mpc_state, last_action):
    open_state = np.zeros((1, 5))
    open_state[0, :4] = mpc_state[:4]
    if last_action[-1] > 0:
        open_state[0, -1] = 0.1
    else:
        open_state[0, -1] = 0
    return open_state.reshape((1, 1, -1))

configuration = {
'experiment_name': 'rndaction_var10',
'pred_model':Dynamic_Base_Model,
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir, #'directory for writing summary.' ,
'num_iterations': 200000,   #'number of training iterations.' ,
'pretrained_model': 'model176002',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'appflow',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_transformed_images': 1,   # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': 1200.0,      # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 16,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
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
'openloop_setup' : setup_openloop_predictor,
'openloop_conf': (IMITATION_BASE_DIR + '/conf_states.py', 'model40000'),
'openloop_conv_state' : state_conv
#'setup_predictor' : setup_predictor
}
