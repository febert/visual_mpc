import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/experiments/cem_exp/benchmarks_goalimage/make_standard_goal_1e4/tfrecords/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

# from video_prediction.prediction_model_downsized_lesslayer import construct_model

configuration = {
'experiment_name': 'rndaction_var10',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'skip_frame': 1,            # 'use ever i-th frame to increase prediction horizon' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 10,      # 'sequence length, including context frames.' ,
'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
'batch_size': 128,           #'batch size for training' ,
'learning_rate': 0.001,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations
'batch_norm': '',
'nomoving_average':'',
'pred_gtruth':''            #feed in both ground-truth and predicted data
}
