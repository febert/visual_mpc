import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-2]) + '/dna_mpc_parallel/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'
LSDC_BASE_DIR = '/'.join(str.split(current_dir, '/')[:-4])

configuration = {
'experiment_name': 'rndaction_var10',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            # name of model weights file from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations

# 'mujoco_file': LSDC_BASE_DIR + '/mjc_models/pushing2d.xml'
'mujoco_file': LSDC_BASE_DIR + '/mjc_models/pushing2d_controller_nomarkers.xml'
}
