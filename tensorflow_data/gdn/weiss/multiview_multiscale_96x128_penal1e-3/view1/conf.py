import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])


from python_visual_mpc.goaldistancenet.variants.multiscale import MultiscaleGoalDistanceNet

import os
# tf record data location:
DATA_DIR = {
    os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/ag_long_records_15kfullres/good': 32,
    os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/ag_long_records_15kfullres/bad': 32,
}

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
'experiment_name': 'correction',
'model':MultiscaleGoalDistanceNet,
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':50000,
'sequence_length':9,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 16,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'orig_size': [96, 128],
'norm':'charbonnier',
'smoothcost':1e-3,
'smoothmode':'2nd',
'flow_penal':1e-3,
'image_only':'',
'ch_mult':4,
'view':1,
'new_loader': True
}