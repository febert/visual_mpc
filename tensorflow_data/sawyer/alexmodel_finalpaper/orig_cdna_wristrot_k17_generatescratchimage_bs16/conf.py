import os
current_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-4]) + '/pushing_data/wrist_rotv1/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

from docile.orig_dna_model import create_model

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model

modelconfiguration = {
'lr': 0.001,
'beta1': 0.9,
'beta2': 0.999,
'l2_weight': 1.0,
'state_weight': 1e-4,
'schedule_sampling_k': 1200,
'context_frames': 2,
'kernel_size': (17, 17),
'layer_normalization': 'in',
'num_transformed_images': 4,
'generate_scratch_image': True,
'vgf_dim': 32,
'trainable_generator_startswith':None,
'dilation_rate': [1, 1],
'model':'orig_cdna',
'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
}

configuration = {
'pred_model':Alex_Interface_Model,
'create_model':create_model,
'context_frames': modelconfiguration['context_frames'],
'experiment_name': 'orig_cdna_wristrot_k17_generatescratchimage_bs16',
'modelconfiguration':modelconfiguration,
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 200000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 48,      # 'sequence length to load, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'schedsamp_k': 900.0,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'kern_size': 17,              #size of DNA kerns
'sawyer':'',
'single_view':"",
'use_len':14,                # number of steps used for training where the starting location is selected randomly within sequencelength
'visual_flowvec':'',
'adim':5,
'sdim':4,
'lstm_size' : np.array([32, 64, 128, 64, 32]),
'num_transformed_images':None,
'normalization':''
}