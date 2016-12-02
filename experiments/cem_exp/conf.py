import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data_files location:
DATA_DIR = '/home/frederik/Documents/pushing_data/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

from video_prediction.prediction_model_downsized_lesslayer import construct_model

configuration = {
'experiment_name': 'cem_control',
'current_dir': current_dir, #'directory for writing gifs' ,
'num_iterations': 100000,   #'number of training iterations.' ,
# 'filepath of a pretrained model to resume training from.' ,
'pretrained_model': '/home/frederik/Documents/lsdc/tensorflow_data/downsized_less_horizon/modeldata/model24002',
'sequence_length': 10,      # 'sequence length, including context frames.' ,
# 'skip_frame': 2,            # 'use ever i-th frame to increase prediction horizon' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 100,           #batch size for evaluation' ,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'downsize': construct_model,           # select the kind of downsized model'
'file_visual': ''          # datafile used for making visualizations
}