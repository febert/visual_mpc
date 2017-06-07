# only using the first pose (code changed accordingly)

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/random_action_var10_pose/train'

lsdc_home = '/'.join(str.split(current_dir, '/')[:-4])
# local output directory
OUT_DIR = current_dir + '/modeldata'

from video_prediction.costmask.setup_predictor_costmask import setup_predictor


configuration = {
'experiment_name': 'costmask',
'setup_predictor': setup_predictor,
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'pretrained_model': lsdc_home + '/tensorflow_data/costmask/moving_retina/modeldata/model48002',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'penal_last_only': False,   # penalize only the last state, to get sharper predictions
'dna_size': 9,              #size of DNA kerns
'use_object_pos':'',
'costmask':'',
'max_move_pos':'',
'moving_retina':'',
'retina_size':25,
'num_obj':4
}