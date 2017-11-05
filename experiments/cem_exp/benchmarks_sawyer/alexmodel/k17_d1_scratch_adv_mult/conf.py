import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import python_visual_mpc
BASE = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model

modelconfiguration = {
'lr': 0.001,
'beta1': 0.9,
'beta2': 0.999,
'l2_weight': 1.0,
'state_weight': 1e-4,
'schedule_sampling_k': -1,
'context_frames': 2,
'model': 'CDNA',
'kernel_size': (17, 17),
'layer_normalization': 'in',
'num_transformed_images': 4,
'generate_scratch_image': True,
'vgf_dim': 32,
'trainable_generator_startswith':None,
'dilation_rate': [1, 1]
}

configuration = {
'context_frames': modelconfiguration['context_frames'],
'setup_predictor': setup_predictor,
'experiment_name': 'improved_cdna_wristrot_k17d1_generatescratchimage_adv_bs16',
'pretrained_model': BASE + '/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_adv_bs16/modeldata/model-205000',
'modelconfiguration':modelconfiguration,
'pred_model':Alex_Interface_Model,
'learning_rate':0.,
'current_dir': current_dir, #'directory for writing summary.' ,
'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 200,           #'batch size for training' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'sawyer':'',
'single_view':"",
'adim':5,
'sdim':4,
}











#
#
# import os
# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# import python_visual_mpc
# BASE = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
# from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor
#
# from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model
#
# configuration = {
# 'experiment_name': 'rndaction_var10',
# 'setup_predictor': setup_predictor,
# 'pred_model':Dynamic_Base_Model,
# 'pretrained_model': BASE + '/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_adv_bs16/modeldata/model-205000',
# 'current_dir': current_dir, #'directory for writing summary.' ,
# 'num_iterations': 200000,   #'number of training iterations.' ,
# 'sequence_length': 15,      # 'sequence length to load, including context frames.' ,
# 'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
# 'context_frames': 2,        # of frames before predictions.' ,
# 'use_state': 1,             #'Whether or not to give the state+action to the model' ,
# 'model': 'cdna',            #'model architecture to use - CDNA, DNA, or STP' ,
# 'num_transformed_images': 4,   # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
# 'schedsamp_k': -1,      # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
# 'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
# 'batch_size': 800,           #'batch size for training' ,
# 'learning_rate': 0.0,     #'the base learning rate of the generator' ,
# 'visualize': '',            #'load model from which to generate visualizations
# 'file_visual': '',          # datafile used for making visualizations
# 'kern_size': 17,             #size of DNA kerns
# 'sawyer':'',
# 'single_view':"",
# 'use_len':14,                # number of steps used for training where the starting location is selected randomly within sequencelength
# 'visual_flowvec':'',
# 'adim':5,
# 'sdim':4,
# 'normalization':'in',
# '1stimg_bckgd':'',
# 'previmg_bckgd':'',
# 'gen_img':'',
# 'ndesig':2
# }
