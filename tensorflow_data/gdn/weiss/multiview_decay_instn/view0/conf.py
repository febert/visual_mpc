import python_visual_mpc
base_dir = python_visual_mpc.__file__

base_dir = '/'.join(str.split(base_dir, '/')[:-2])

# tf record data location:
import os
DATA_DIR = {os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/sudri_ag/good': 4,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/sudri_ag/bad': 6,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/vestri_ag/good': 4,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/vestri_ag/bad': 6,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/sudri_ag_long/good': 6,
         os.environ['VMPC_DATA_DIR'] + '/sawyer_grasping/sawyer_data/sudri_ag_long/bad': 6,
            }
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'


configuration = {
'experiment_name': 'correction',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': base_dir,   #'directory for writing summary.' ,
'num_iterations':200000,
'sequence_length':15,
'visualize':'',
'skip_frame':1,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'normalization':'in',
'sdim' :5,
'adim' :4,
'orig_size': [48,64],
'norm':'charbonnier',
'smoothcost':1e-6,
'smoothmode':'2nd',
'fwd_bwd':'',
'flow_diff_cost':1e-4,
'hard_occ_thresh':'',
'occlusion_handling':1e-4,
'occ_thres_mult':0.5,
'occ_thres_offset':1.,
'flow_penal':1e-4,
'ch_mult':4,
'view':1,
'decay_lr':''
}