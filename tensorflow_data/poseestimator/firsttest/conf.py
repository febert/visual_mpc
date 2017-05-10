import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/large_displacement_pose/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

# from video_prediction.prediction_model_downsized_lesslayer import construct_model

configuration = {
'experiment_name': 'rndaction_var10',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'num_iterations': 100000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 32,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'use_object_pos':'',
'batch_norm': '',
'nomoving_average':'',
}
