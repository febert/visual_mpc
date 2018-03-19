import os
from python_visual_mpc.imitation_model.imitation_model import ImitationBaseModel

DATA_DIR = os.environ['VMPC_DATA_DIR']

configuration = {
    'model' : ImitationBaseModel,
    'data_dir':DATA_DIR  + '/cartgripper_det_grasp/train/',
    'model_dir':DATA_DIR + '/cartgripper_det_grasp/trained_model/',
    'n_iters':50000,
    'n_print':100,
    'n_save':500,
    'learning_rate':0.001,
    'train_val_split':0.95,
    'adim':5,
    'sdim':12,
    'orig_size': [48,64],
    'skip_frame' : 1,
    'sequence_length' : 15,
    'batch_size' : 64,
    'vgg19_path': DATA_DIR,
    'MDN_loss' : 3
}


