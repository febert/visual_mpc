import os

DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper_det_grasp/train/'

configuration = {
    'data_dir':DATA_DIR,
    'train_val_split':0.95,
    'adim':5,
    'sdim':12,
    'orig_size': [48,64],
    'skip_frame' : 1,
    'sequence_length' : 15,
    'batch_size' : 64,
    'vgg19_path': os.path.expanduser('~/Documents/visual_mpc/pushing_data/')
}


