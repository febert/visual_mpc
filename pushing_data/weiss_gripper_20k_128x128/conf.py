import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/weiss_gripper"

configuration = {
'tf_rec_dir': current_dir + '/train',      #'directory for model checkpoints.' ,
'source_basedirs': [source_basedir + '/vestri/main', source_basedir + '/austri/main'],
'sourcetags': ['/main'],               # list of source dirs for different camera view-points
'total_num_img': 96,
'take_ev_nth_step':2,                          # subsample trajectories
'target_res': (128, 128),                             #128x128
'shrink_before_crop':1/4.5,
'ngroup':1000,
'rowstart':20,
'colstart':64,
'adim':5,
'sdim':4,
'brightness_threshold':int(0.25*255.),            # if average pixel value lower discard video
'tf_rec_dir': os.environ['VMPC_DATA_DIR'] + '/weiss_gripper_20k_128x128/train'
}