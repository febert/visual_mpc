import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/wrist_rot"

configuration = {
'tf_rec_dir': current_dir + '/train',      #'directory for model checkpoints.' ,
'source_basedir': source_basedir,
'sourcedirs': [source_basedir + '/main'],               # list of source dirs for different camera view-points
'total_num_img': 96,
'take_ev_nth_step':2,                          # subsample trajectories
'target_res': (128,128),                             #128x128
'shrink_before_crop':1/5.,
'rowstart':10,
'colstart':45,
'adim':5,
'sdim':4,
}

assert configuration['shrink_before_crop'] != 0