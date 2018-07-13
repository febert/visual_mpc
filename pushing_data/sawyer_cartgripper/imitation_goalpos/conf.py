import os


from python_visual_mpc.imitation_model.openloop_models.goal_image_models import AttentionGoalImages

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
DATA_DIR = os.environ['VMPC_DATA_DIR']
configuration = {
    'model' :AttentionGoalImages,
    'data_dir':BASE_DIR  + '/data/',
    'model_dir':BASE_DIR + '/modeldata/',
    'ncam' : 2,
    'n_iters':5000,
    'n_print':100,
    'n_save':1000,
    'clip_grad':5.,
    'learning_rate':1e-3,
    'train_val_split':0.95,
    'num_feats' : 16,
    'adim':4,
    'sdim':5,
    'orig_size': [48,64],
    'skip_frame' : 1,
    'sequence_length' : 17,
    'batch_size' : 64,
    'vgg19_path': DATA_DIR,
    'MDN_loss' : 20,
    'N_GEN' : 20000,
    'color_augmentation' : ''
}

