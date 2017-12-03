import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/online_data"

configuration = {
'current_dir':current_dir,
'tf_rec_dir': current_dir + '/train',      #'directory for model checkpoints.' ,
'source_basedirs': [source_basedir+'/source0', source_basedir+'/source1'],
'sourcetags': ['/main'],               # list of source dirs for different camera view-points
'total_num_img': 56,
'take_ev_nth_step': 4,                          # subsample trajectories
'target_res': (64,64),                             #128x128
'shrink_before_crop':1/9.,
'rowstart':10,
'colstart':32,
'adim':5,
'sdim':4,
'brightness_threshold':int(0.25*255.),            # if average pixel value lower discard video
'allow_totalnumimg_smaller_numactions':""
}