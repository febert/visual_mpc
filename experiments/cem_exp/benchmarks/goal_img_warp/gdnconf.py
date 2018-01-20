import python_visual_mpc
base_dir = python_visual_mpc.__file__
base_dir = '/'.join(str.split(base_dir, '/')[:-2])

# tf record data location:
DATA_DIR = base_dir + '/pushing_data/cartgripper_vidpred/train'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
'experiment_name': 'correction',
'pretrained_model': base_dir + '/tensorflow_data/gdn/vidpred_data/modeldata/model2',  ##########!!!!!!!!!!!!
'data_dir': '',       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':50000,
'sequence_length':14,
'visualize':'',
'skip_frame':1,
'num_epochs': 40,   #'number of training iterations.' ,
'batch_size': 200,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'orig_size': [48,64],
'vidpred_data':''           # tell loader to get video prediction data
}
