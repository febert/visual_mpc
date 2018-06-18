
import cv2
import numpy as np
import pickle as pkl
import re
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
import os
import python_visual_mpc

import glob

ROOT_DIR = os.path.abspath(python_visual_mpc.__file__)
ROOT_DIR = '/'.join(str.split(ROOT_DIR, '/')[:-2])


def get_folders(dir):
    files = glob.glob(dir + '/*')
    folders = []
    for f in files:
        if os.path.isdir(f):
            folders.append(f)
    return sorted_alphanumeric(folders)


def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def annotate(bench_dir):

    scores_l = []
    improvements_l = []
    initial_dists_l = []
    names = []

    bench_dir = bench_dir + '/bench'

    folders = get_folders(bench_dir)

    for folder in folders:
        exp_dir = folder + '/videos'
        name = str.split(folder, '/')[-1]
        print('example: ', name)

        if os.path.exists(exp_dir + '/desig_goal_pixstart.pkl'):
            desig_pix_t0 = pkl.load(open(exp_dir + '/desig_goal_pixstart.pkl', 'rb'))['desig_pix'][0]
            goal_pix = pkl.load(open(exp_dir + '/desig_goal_pixgoal.pkl', 'rb'))['desig_pix'][0]
        else:
            dict = pkl.load(open(exp_dir + '/desig_goal_pixstart_traj0.pkl', 'rb'))
            desig_pix_t0 = dict['desig_pix'][0]
            goal_pix = dict['goal_pix'][0]


        # if os.path.exists(exp_dir + '/img_goal.png'):
        #     start_image = cv2.imread(exp_dir + '/img_start.png')[:, :, ::-1]
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(start_image)
        #     plt.title('start_image')
        #
        #     goal_image = cv2.imread(exp_dir + '/img_goal.png')[:, :, ::-1]
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(goal_image)
        #     plt.title('goal_image')
        #     plt.show()
        # else:
        #     goal_image = cv2.imread(exp_dir + '/img_start_traj0.png')[:, :, ::-1]
        #     plt.imshow(goal_image)
        #     plt.title('goal_image')
        #     plt.show()

        final_image = cv2.imread(exp_dir + '/finalimage.png')[:,:,::-1]

        c = Getdesig(final_image)
        final_pos = c.coords

        final_dist = np.linalg.norm(goal_pix - final_pos)
        initial_dist = np.linalg.norm(desig_pix_t0 - goal_pix)
        improvement = initial_dist - final_dist
        print('initial_dist {}  finaldist {} improvement {} '.format(initial_dist, final_dist, improvement))

        scores_l.append(final_dist)
        improvements_l.append(improvement)
        initial_dists_l.append(initial_dist)
        names.append(name)

        ann_stats = {'names':np.array(names),
                     'initial_dist': np.array(initial_dists_l),
                     'improvement': np.array(improvements_l),
                     'scores': np.array(scores_l)}
        pkl.dump(ann_stats, open(bench_dir + '/ann_stats.pkl','wb'))

        write(bench_dir, ann_stats)

def write(exp_dir, stat):
    improvement = stat['improvement']

    scores = stat['scores']
    if 'initial_dist' in stat:
        initial_dist = stat['initial_dist']
    else:
        initial_dist = None

    sorted_ind = improvement.argsort()[::-1]

    mean_imp = np.mean(improvement)
    med_imp = np.median(improvement)
    mean_dist = np.mean(scores)
    med_dist = np.median(scores)

    result_file = exp_dir + '/result.txt'
    f = open(result_file, 'w')
    f.write('---\n')
    f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos improvemnt: {0}\n'.format(mean_imp))
    f.write('median pos improvement {}\n'.format(med_imp))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement) / np.sqrt(improvement.shape[0])))
    f.write('---\n')
    f.write('average pos score: {0}\n'.format(mean_dist))
    f.write('median pos score {} \n'.format(med_dist))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(scores) / np.sqrt(scores.shape[0])))
    f.write('---\n')
    f.write('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))
    f.write('---\n')
    f.write('average initial dist: {0}\n'.format(np.mean(initial_dist)))
    f.write('median initial dist: {0}\n'.format(np.median(initial_dist)))
    f.write('----------------------\n')
    f.write('traj: improv, score, term_t, lifted, rank\n')
    f.write('----------------------\n')

    for n in range(improvement.shape[0]):
        f.write('{}: {}, {}:{}\n'.format(n, improvement[n], scores[n], np.where(sorted_ind == n)[0][0]))
    f.close()


if __name__ == '__main__':
    # path = '/home/febert/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_regstartgoal_tradeoff'
    path = '/home/febert/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_dynrnn'
    annotate(path)

    # pkl.load(open(bench_dir + '/ann_stats.pkl', 'wb'))
