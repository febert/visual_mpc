import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-4]) + '/pushing_data/wrist_rotv1/train'
TEST_DATA_DIR = '/'.join(str.split(current_dir, '/')[:-4]) + '/pushing_data/wristrot_test_seenobj/test'

# local output directory
OUT_DIR = current_dir + '/modeldata'


from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model

modelconfiguration = {
'lr': 0.001,
'beta1': 0.9,
'beta2': 0.999,
'l2_weight': 1.0,
'state_weight': 1e-4,
'schedule_sampling_k': 1200,
'context_frames': 2,
'model': 'CDNA',
'kernel_size': (17, 17),
'layer_normalization': 'in',
'num_transformed_images': 4,
'generate_scratch_image': True,
'vgf_dim': 32,
'trainable_generator_startswith':None,
'dilation_rate': [1, 1],
'ndesig':1,
}

configuration = {
'context_frames': modelconfiguration['context_frames'],
'experiment_name': 'improved_cdna_wristrot_k17d1_generatescratchimage_bs16',
'modelconfiguration':modelconfiguration,
'pred_model':Alex_Interface_Model,
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'test_data_dir': TEST_DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir, #'directory for writing summary.' ,
'num_iterations': 200000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 48,      # 'sequence length to load, including context frames.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 16,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'kern_size': 9,             #size of DNA kerns
'sawyer':'',
'single_view':"",
'use_len':14,                # number of steps used for training where the starting location is selected randomly within sequencelength
'adim':5,
'sdim':4,
'normalization':'in',
'gen_img':'',
'ndesig':modelconfiguration['ndesig']
}


