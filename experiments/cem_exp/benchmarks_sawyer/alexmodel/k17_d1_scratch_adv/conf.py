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
