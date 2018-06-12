import os

DATA_DIR = os.environ['VMPC_DATA_DIR']
from python_visual_mpc.imitation_model.openloop_models.base_model import LSTMAttentionOpenLoop
configuration = {
    'model' : LSTMAttentionOpenLoop,
    'data_dir':DATA_DIR  + '/cartgripper_det_grasp/train/',
    'model_dir':DATA_DIR + '/cartgripper_imitation_openloop/LSTMattention_mdn_states/',
    'n_iters':200000,
    'n_print':100,
    'n_save':10000,
    'clip_grad':5.,
    'learning_rate':1e-3,
    'train_val_split':0.95,
    'adim':5,
    'sdim':5,
    'orig_size': [48,64],
    'skip_frame' : 1,
    'sequence_length' : 15,
    'batch_size' : 64,
    'vgg19_path': DATA_DIR,
    'MDN_loss' : 4,
    'nactions' : 5,
    'num_repeat': 3,
    'lstmforward_dim':256,
    'num_heads' : 4,
    'N_GEN' : 20000
}

