import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir = "/mnt/sda1/sawyerdata/softmotion_0511"

configuration = {
'tf_rec_dir': current_dir + '/train',      #'directory for model checkpoints.' ,
'source_basedir': source_basedir,
'sourcedirs': [source_basedir + '/aux1'],               # list of source dirs for different camera view-points
'pkl_source':'main',            # if the pkl file is in a different source tree
'total_num_img': 30,
'take_ev_nth_step':1,                          # subsample trajectories
'target_res': (64,64),                             #128x128
'crop_before_shrink':'',
'rowstart':60,
'colstart':160,
'raw_image_height':750,
'adim':4,
'sdim':3,
'imagename_no_zfill':'',            #don't pad imagefile index with zeros
}