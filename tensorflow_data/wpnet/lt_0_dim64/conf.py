import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
'experiment_name': 'correction',
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'data_dir': DATA_DIR,      # 'directory containing data.' ,'
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':100000,
'sequence_length':3,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 16,           #'batch size for training' ,
'learning_rate': 0.0001,     #'the base learning rate of the generator' ,
'normalization':'None',
'tweights_reg':0.,
'orig_size':[48, 64],
'lt_cost_factor':0.,
'lt_dim':64,
'sdim':6,
'adim':3,
}