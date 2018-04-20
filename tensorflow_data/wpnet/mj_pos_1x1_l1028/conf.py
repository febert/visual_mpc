import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])


import os
# tf record data location:
DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/mj_pos_noreplan_fast_tfrec/train'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))


# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
    'experiment_name': 'correction',
    'data_dir': DATA_DIR,      # 'directory containing data.' ,'
    'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
    'current_dir': base_dir,   #'directory for writing summary.' ,
    'num_iterations':100000,
    'sequence_length':15,
    'train_val_split':.95,
    'visualize':'',
    'skip_frame':1,
    'batch_size': 16,           #'batch size for training' ,
    'learning_rate': 0.0001,     #'the base learning rate of the generator' ,
    'normalization':'bnorm',
    'tweights_reg':0.,
    'orig_size':[48, 64],
    'lt_cost_factor':1e-6,
    'lt_dim': 1028,
    'sched_lt_cost':[30000, 60000],
    'sdim':12,
    'adim':5,
    'MSE':'',
    'enc_avg_pool':[1,1],
    'inference_use_cont':'',
}