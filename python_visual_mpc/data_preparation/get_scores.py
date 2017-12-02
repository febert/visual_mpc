import cPickle
import copy
import glob
import os
import random

import imutils  # pip install imutils
import numpy as np
import tensorflow as tf
from PIL import Image

from python_visual_mpc.data_preparation.gather_data import make_traj_name_list
import cv2
import ray
import create_gif
import argparse
import sys
import imp

import re


def get_scores(traj_name_list):
    nopkl_file = 0

    for trajname in traj_name_list:  # loop of traj0, traj1,..

        # print 'processing {}, seq-part {}'.format(trajname, traj_tuple[1] )
        # try:
        traj_index = re.match('.*?([0-9]+)$', trajname).group(1)

        traj_tailpath = '/'.join(str.split(trajname, '/')[-2:])
        traj_beginpath = '/'.join(str.split(trajname, '/')[:-3])

        pkl_file = trajname + '/joint_angles_traj{}.pkl'.format(traj_index)
        if not os.path.isfile(pkl_file):
            nopkl_file += 1
            print 'no pkl file found, file no: ', nopkl_file
            continue

        pkldata = cPickle.load(open(pkl_file, "rb"))
        track = pkldata['track_desig']
        track = pkldata['']



    print 'done, number of pkl files not found:', nopkl_file



def main():
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('hyper', type=str, help='configuration file name')
    args = parser.parse_args()

    conf_file = args.hyper
    if not os.path.exists(args.hyper):
        sys.exit("configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    traj_name_list = make_traj_name_list(conf)


    get_scores(traj_name_list)


if __name__ == "__main__":
    # make_train_test_split()
    main()