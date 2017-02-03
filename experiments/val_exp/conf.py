import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import getpass
getpass.getuser()

from video_prediction.setup_predictor_stochastic_search import setup_predictor

configuration = {
'experiment_name': 'cem_control',
'current_dir': current_dir, #'directory for writing gifs' ,
# 'filepath of a pretrained model to use for cem
'pretrained_model': '/home/{}/Documents/lsdc/tensorflow_data/stochastic_search_multgpu/modeldata/model36002'.format(getpass.getuser()),
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 204,           #batch size for evaluation' ,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'setup_predictor': setup_predictor,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations,
'penal_last_only': False,
'noise_dim': 5,
'no_imagerepeat': True   # comment this out to deactivate
}


# configuration = {
# 'experiment_name': 'cem_control',
# 'current_dir': current_dir, #'directory for writing gifs' ,
# 'num_iterations': 100000,   #'number of training iterations.' ,
# # 'filepath of a pretrained model to use for cem
# 'pretrained_model': '/home/frederik/Documents/lsdc/tensorflow_data/random_actions_var10/modeldata/model48002',
# 'sequence_length': 15,      # 'sequence length, including context frames.' ,
# 'context_frames': 2,        # of frames before predictions.' ,
# 'use_state': 1,             #'Whether or not to give the state+action to the model' ,
# 'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
# 'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
# 'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
# 'batch_size': 3,           #batch size for evaluation' ,
# 'learning_rate': 0,     #'the base learning rate of the generator' ,
# 'visualize': '',            #'load model from which to generate visualizations
# 'downsize': construct_model,           # select the kind of downsized model'
# 'file_visual': '',          # datafile used for making visualizations,
# 'penal_last_only': True
# }