import os

DATA_DIR = os.environ['VMPC_DATA_DIR']
from python_visual_mpc.imitation_model.imitation_model import ImitationLSTMModelStateGoalImage
configuration = {
    'model' : ImitationLSTMModelStateGoalImage,
    'data_dir':DATA_DIR  + '/cartgripper_imitation_goalpos/train/',
    'model_dir':DATA_DIR + '/cartgripper_imitation_goalpos/lstm_model_mdn_states_slr/',
    'n_iters':80000,
    'n_print':100,
    'n_save':500,
    'clip_grad':5.,
    'learning_rate':1e-3,
    'train_val_split':0.95,
    'adim':5,
    'sdim':5,
    'goal_image' : True,
    'orig_size': [48,64],
    'skip_frame' : 1,
    'sequence_length' : 15,
    'batch_size' : 64,
    'vgg19_path': DATA_DIR,
    'MDN_loss' : 20,
    'lstm_layers':[128, 128, 128],
    'N_GEN' : 1000
}

