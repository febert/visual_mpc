import pickle
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
# import ray
from . import create_gif
import argparse
import sys
import imp

import re

FINAL_WEIGHT = 10.

def get_scores(conf, traj_name_list):
    nopkl_file = 0

    eval_files = []
    scores = []
    start_dist  =[]
    final_dist = []
    final_improvement = []


    for trajname in traj_name_list:  # loop of traj0, traj1,..

        # print 'processing {}, seq-part {}'.format(trajname, traj_tuple[1] )
        # try:
        traj_index = re.match('.*?([0-9]+)$', trajname).group(1)

        pkl_file = trajname + '/joint_angles_traj{}.pkl'.format(traj_index)
        if not os.path.isfile(pkl_file):
            nopkl_file += 1
            print('no pkl file found, file no: ', nopkl_file)
            continue

        pkldata = pickle.load(open(pkl_file, "rb"))
        if 'track_desig' in pkldata and 'goal_pos' in pkldata:
            print('processing', trajname)

            track = np.squeeze(pkldata['track_desig'])
            goal_pos = np.squeeze(pkldata['goal_pos'])


            tlen = track.shape[0]

            dist_t = [np.linalg.norm(goal_pos - track[t]) for t in range(tlen)]
            score = 0
            for t, dist in enumerate(dist_t):
                if t == len(dist_t)-1:
                    factor = FINAL_WEIGHT
                else:
                    factor = 1.
                score += (dist_t[0] - dist)*factor
            score /= tlen

            scores.append(score)
            start_dist.append(dist_t[0])
            final_dist.append(dist_t[-1])
            final_improvement.append(dist_t[0] - dist_t[-1])
            eval_files.append(trajname)

    avg_start_dist = np.mean(np.array(start_dist))
    avg_final_dist = np.mean(np.array(final_dist))
    avg_final_improvement = np.mean(np.array(final_improvement))
    avg_score = np.mean(np.array(scores))

    n = len(start_dist)
    std_start_dist = np.std(np.array(start_dist))/np.sqrt(n)
    std_final_dist = np.mean(np.array(final_dist))/np.sqrt(n)
    std_final_improvement = np.mean(np.array(final_improvement))/np.sqrt(n)
    std_score = np.mean(np.array(scores))/np.sqrt(n)

    file = conf['current_dir'] +'/results_summary.txt'
    print('writing:', file)
    with open(file, 'w+') as f:
        f.write('evaluated {} trajectories \n'.format(n))
        f.write('average start distances: {} std. err {}\n'.format(avg_start_dist, std_start_dist))
        f.write('average final distances: {}  std. err {}\n'.format(avg_final_dist, std_final_dist))
        f.write('average final improvement: {} std. err {}\n'.format(avg_final_improvement, std_final_improvement))
        f.write('average score: {} std. err {}  (calculated with finalweight {})\n'.format(avg_score, std_score, FINAL_WEIGHT))

    file = conf['current_dir'] + '/per_traj_scores.txt'
    print('writing:', file)
    with open(file, 'w+') as f:
        f.write('file, start distances, final distances, final improvement, score \n'.format(avg_start_dist))
        for i, name in enumerate(eval_files):
            f.write('{}:  {}; {}; {}; {}\n'.format(name, start_dist[i], final_dist[i], final_improvement[i], scores[i]))

    print('done, number of pkl files not found:', nopkl_file)

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('hyper', type=str, help='configuration file name')
    args = parser.parse_args()

    conf_file = args.hyper
    if not os.path.exists(args.hyper):
        sys.exit("configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    traj_name_list = make_traj_name_list(conf, shuffle=False)
    get_scores(conf, traj_name_list)


if __name__ == "__main__":
    # make_train_test_split()
    main()