import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/weiss_gripper"

configuration = {
'tf_rec_dir': current_dir + '/train',      #'directory for model checkpoints.' ,
'source_basedirs': [source_basedir + '/vestri', source_basedir + '/austri'],
'sourcetags': ['/main'],               # list of source dirs for different camera view-points
'total_num_img': 96,
'take_ev_nth_step':2,                          # subsample trajectories
'target_res': (64,64),                             #128x128
'shrink_before_crop':1/11.,
'rowstart':6,
'colstart':22,
'adim':5,
'sdim':4,
'brightness_threshold':int(0.25*255.),            # if average pixel value lower discard video
}



