# data loading configuration

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# explanation of source_tag dictionary:
# name: the name of the data field
# dtype: the dtype
# shape: the target shape, will be cropped to match this shape
# rowstart: starting row for cropping
# rowend: end row for cropping
# colstart: start column for cropping
# shrink_before_crop: shrink image according to this ratio before cropping
# brightness_threshold: if average pixel value lower discard video

source_tag_images = {'name': 'images',
                     'file':'/images/im{}.png',   # only tindex
                     'shape':[48,64,3],
                     'rowstart':15,
                     'colstart':0,
                     }

source_tag_states = {'name': 'states',
                     'shape':[6],
                     'file':'/state_action.pkl',
                     'pkl_names':['qpos','qvel']}

source_tag_actions = {'name':'actions',
                      'shape': [3],
                      'file':'/state_action.pkl'
                      }

sequence_length = 15
take_ev_nth_step = 1
total_num_img = sequence_length*take_ev_nth_step

configuration = {
'current_dir':current_dir,
'source_basedirs': [current_dir +'/test'],
# 'sourcetags': [source_tag_images],               # list of source dirs for different camera view-points
'sourcetags': [source_tag_images, source_tag_states, source_tag_actions],               # list of source dirs for different camera view-points
'pkl_file':'/state_action.pkl',                 #pkl file found in trajectory folder
'sequence_length':sequence_length,
'take_ev_nth_step': take_ev_nth_step,                          # subsample trajectories
'total_num_img': total_num_img,                 # total number of images in folders
'ngroup':1000
}