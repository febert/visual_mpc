import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-4]) + '/pushing_data/force_ctrl_save_pos_correction/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

from video_prediction.prediction_model_downsized_lesslayer import construct_model

configuration = {
'experiment_name': 'rndaction_var10',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'num_objects': 4,
'use_object_pos': True
}
