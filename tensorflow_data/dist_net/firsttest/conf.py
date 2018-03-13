import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

import os

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
    'experiment_name': 'correction',
    'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
    'current_dir': base_dir,   #'directory for writing summary.' ,
    'num_iterations':100000,
    'sequence_length':30,
    'train_val_split':.95,
    'visualize':'',
    'skip_frame':1,
    'batch_size': 16,           #'batch size for training' ,
    'learning_rate': 0.0001,     #'the base learning rate of the generator' ,
    'normalization':'bnorm',
    'source_basedirs':[os.environ["VMPC_DATA_DIR"] + '/datacol_appflow/data/train'],
    'orig_size':[48, 64],
    'adim':3,
    'sdim':6
}