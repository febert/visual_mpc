import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
BASE_DIR = '/'.join(str.split(current_dir, '/')[:-4])

DATA_DIR = BASE_DIR + '/pushing_data/random_action_var10/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

from reward_network.setup_rewardnet import setup_rewardnet

configuration = {
'experiment_name': 'rndaction_var10',
'setup_rewardnet': setup_rewardnet,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'pretrained_model': BASE_DIR + '/tensorflow_data/rewardnet/large_displacement_finalimage/modeldata/model48002',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 200,           #'batch size for training' ,
'learning_rate': 0.0,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'batch_norm': '',
'nomoving_average':''
}
