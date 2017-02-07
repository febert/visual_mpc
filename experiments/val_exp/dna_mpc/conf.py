import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import getpass
getpass.getuser()

from video_prediction.prediction_model_downsized_lesslayer import construct_model

configuration = {
'experiment_name': 'cem_control',
'current_dir': current_dir, #'directory for writing gifs' ,
# 'filepath of a pretrained model to use for cem
'pretrained_model': '/home/{}/Documents/lsdc/tensorflow_data/dna/modeldata/model48002'.format(getpass.getuser()),
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 200,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'downsize': construct_model,           # select the kind of downsized model'
'file_visual': '',          # datafile used for making visualizations,
'penal_last_only': False,
}
