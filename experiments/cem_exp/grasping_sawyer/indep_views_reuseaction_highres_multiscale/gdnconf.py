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

configuration = {
'experiment_name': 'correction',
'pred_model':MulltiviewTestGDN,
'pretrained_model': [base_dir + '/tensorflow_data/gdn/weiss/multiview_multiscale_96x128_highpenal/view0/modeldata/model28002',
                     base_dir + '/tensorflow_data/gdn/weiss/multiview_multiscale_96x128_highpenal/view1/modeldata/model28002'],
'batch_size':1,
'current_dir': base_dir,   #'directory for writing summary.' ,
'normalization':'None',
'orig_size': [96, 128],
'norm':'charbonnier',
'smoothcost':1e-5,
'smoothmode':'2nd',
'flow_penal':1e-4,
'image_only':'',
'ch_mult':4,
'multi_scale':'',
'view':0,
'new_loader': True
}


