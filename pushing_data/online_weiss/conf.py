import os
current_dir = os.path.dirname(os.path.realpath(__file__))

source_basedir0 = "/mnt/sda1/sawyerdata/wrist_rot"

data_configuration = {
    'current_dir':current_dir,
    'source_basedirs': [source_basedir0],
    'sourcetags': ['/main'],  # list of source dirs for different camera view-points
    'total_num_img': 96,
    'take_ev_nth_step': 2,  # subsample trajectories
    'target_res': (64, 64),  # 128x128
    'shrink_before_crop': 1 / 9.,
    'rowstart': 10,
    'colstart': 32,
    'adim': 5,
    'sdim': 4,
}