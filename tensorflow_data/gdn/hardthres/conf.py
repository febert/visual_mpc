import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

# tf record data location:
DATA_DIR = base_dir + '/pushing_data/cartgripper/train'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'

tag_images = {'name': 'images',
             'file':'/images/im{}.png',   # only tindex
             'shape':[48, 64,3],
               }

configuration = {
'experiment_name': 'correction',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':50000,
'sequence_length':14,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 64,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'sdim' :6,
'adim' :3,
'orig_size': [48,64],
'norm':'charbonnier',
'smoothcost':1e-6,
'smoothmode':'2nd',
'fwd_bwd':'',
'flow_diff_cost':1e-4,
'hard_occ_thresh':'',
'occlusion_handling':1e-4,

'ngroup':100,
'sourcetags':[tag_images],
'source_basedirs':[base_dir + '/pushing_data/cartgripper_startgoal/train'],
}