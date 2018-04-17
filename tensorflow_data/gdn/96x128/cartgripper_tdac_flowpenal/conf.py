import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])


import os
# tf record data location:
DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper_updown_sact_96x128/train'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

from python_visual_mpc.goaldistancenet.variants.temp_dividenconquer import Temp_DnC_GDnet

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
'model':Temp_DnC_GDnet,
'experiment_name': 'correction',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':50000,
'sequence_length':5,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 64,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'orig_size': [96,128],
'norm':'charbonnier',
'smoothcost':1e-6,
'smoothmode':'2nd',
'image_only':'',
'ch_mult':1,
'temp_divide_and_conquer':'',
'cons_loss':0.,
'flow_penal':1e-4,
}