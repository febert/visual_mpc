import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

# tf record data location:
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'
# local output directory

from python_visual_mpc.goaldistancenet.multiview_testgdn import MulltiviewTestGDN
from python_visual_mpc.goaldistancenet.variants.multiscale import MultiscaleGoalDistanceNet

configuration = {
'experiment_name': 'correction',
'pred_model':MulltiviewTestGDN,
'model':MultiscaleGoalDistanceNet,
'pretrained_model': [base_dir + '/tensorflow_data/gdn/weiss/multiview_multiscale_96x128_highpenal/view0/modeldata/model48002',
                     base_dir + '/tensorflow_data/gdn/weiss/multiview_multiscale_96x128_highpenal/view1/modeldata/model48002'],
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':50000,
'sequence_length':9,
'train_val_split':.95,
'visualize':'',
'skip_frame':1,
'batch_size': 1,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'None',
'orig_size': [96, 128],
'norm':'charbonnier',
'smoothcost':1e-5,
'smoothmode':'2nd',
'flow_penal':1e-4,
'image_only':'',
'ch_mult':4,
'view':0,
'new_loader': True
}


