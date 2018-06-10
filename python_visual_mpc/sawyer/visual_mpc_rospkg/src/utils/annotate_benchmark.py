
import cv2
import numpy as np
import pickle as pkl

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
import os
import python_visual_mpc

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])

def annotate(exp_dir):
    num_runs = 1

    scores = []
    improvements = []

    for n in range(num_runs):
        exp_dir + '/videos/' +

        desig_pix_t0 = pkl.load(open('points.pkl', 'rb'))
        goal_pix = pkl.load(open('points.pkl', 'rb'))

        start_image =

        goal_image =
        final_image = cv2.imread()

        c = Getdesig(final_image)
        final_pos = np.round(c.coords).astype(np.int)

        final_dist = np.linalg.norm(goal_pix - final_pos)
        improvement = np.linalg.norm(desig_pix_t0 - goal_pix) - final_dist
        scores.append(final_dist)
        improvements.append(improvement)









def read_data(exp_dir)

    for ex in exmps:
        print(ex)
        c = Getdesig(goal_images[ex])
        goal_pix.append(np.round(c.coords).astype(np.int))
        # goal_pix.append(np.array([0,0]))
        annotated_goal_images.append(add_crosshairs_single(goal_images[ex], goal_pix[-1]))


def cal





if __name__ == '__main__':
    read_data()
