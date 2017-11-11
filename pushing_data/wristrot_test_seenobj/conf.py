import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/wristrot_test_seenobj"

configuration = {
'tf_rec_dir': current_dir + '/test',      #'directory for model checkpoints.' ,
'source_basedir': source_basedir,
'sourcedirs': [source_basedir + '/main'],               # list of source dirs for different camera view-points
'total_num_img': 96,
'take_ev_nth_step':2,                          # subsample trajectories
'target_res': (64,64),                             #128x128
'shrink_before_crop':1/9.,
'rowstart':10,
'colstart':32,
'adim':5,
'sdim':4,
}