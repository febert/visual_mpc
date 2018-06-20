from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
import os
import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from python_visual_mpc.utils.txt_in_image import draw_text_onimage
import sys
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import imp
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.benckmarking.annotate_benchmark import get_folders
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single
from PIL import Image
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig

save_points = False
FLAGS = flags.FLAGS

def load_data(bench_dir, tsteps, num_ex=8, grasp_data_mode=-1, interval=12):

    folders = get_folders(bench_dir)
    # folders = [bench_dir + '/ob18s1']
    folders = folders[:num_ex]

    image_batch = []
    desig_pix_t0_l = []
    goal_pix_l = []

    if grasp_data_mode != -1 and not save_points:
        dict = pkl.load(open(bench_dir + '/points{}.pkl'.format(grasp_data_mode), 'rb'))
        desig_pix_t0 = dict['desig_pix_t0']
        goal_pix = dict['goal_pix']

    for folder in folders:
        name = str.split(folder, '/')[-1]
        print('example: ', name)
        if grasp_data_mode != -1:
            exp_dir = folder + '/traj_data/images{}'.format(grasp_data_mode)
        else:
            exp_dir = folder + '/videos'
            desig_pix_t0 = pkl.load(open(exp_dir + '/desig_goal_pixstart.pkl', 'rb'), encoding='latin1')['desig_pix'][0]
            goal_pix = pkl.load(open(exp_dir + '/desig_goal_pixgoal.pkl', 'rb'), encoding='latin1')['desig_pix'][0]
            desig_pix_t0_l.append(desig_pix_t0.astype(np.int))
            goal_pix_l.append(goal_pix.astype(np.int))

        imlist = []
        for t in range(0, tsteps, interval):
            if grasp_data_mode != -1:
                name = exp_dir + '/im{}.png'.format(t)
                imlist.append(np.array(Image.open(name)))
            else:
                name = exp_dir + '/small{}.jpg'.format(t)
                imlist.append(np.array(Image.open(name)))

        if grasp_data_mode != -1 and save_points:
            plt.switch_backend('TkAgg')
            c_main = Getdesig(imlist[0])
            desig_pix_t0_l.append(c_main.coords.astype(np.int64))
            c_main = Getdesig(imlist[-1])
            goal_pix_l.append(c_main.coords.astype(np.int64))

        images = np.stack(imlist).astype(np.float32)/255.
        image_batch.append(images)

    image_batch = np.stack(image_batch)

    if desig_pix_t0_l != []:
        desig_pix_t0 = np.stack(desig_pix_t0_l)
        goal_pix = np.stack(goal_pix_l)

    if save_points:
        pkl.dump({'desig_pix_t0':desig_pix_t0, 'goal_pix':goal_pix} , open(bench_dir + '/points{}.pkl'.format(grasp_data_mode), 'wb'))

    return image_batch, desig_pix_t0, goal_pix

def load_benchmark_data():
    folders = get_folders(bench_dir)
    # folders = [bench_dir + '/ob18s1']
    folders = folders[:num_ex]

    image_batch = []
    desig_pix_t0_l = []
    goal_pix_l = []


    dict = pkl.load(open(bench_dir + '/points{}.pkl'.format(grasp_data_mode), 'rb'))
    desig_pix_t0 = dict['desig_pix_t0']
    goal_pix = dict['goal_pix']

    for folder in folders:
        name = str.split(folder, '/')[-1]
        print('example: ', name)
        if grasp_data_mode != -1:
            exp_dir = folder + '/traj_data/images{}'.format(grasp_data_mode)

        imlist = []
        for t in range(0, tsteps, interval):
            if grasp_data_mode != -1:
                name = exp_dir + '/im{}.png'.format(t)
                imlist.append(np.array(Image.open(name)))
            else:
                name = exp_dir + '/small{}.jpg'.format(t)
                imlist.append(np.array(Image.open(name)))

        if grasp_data_mode != -1 and save_points:
            plt.switch_backend('TkAgg')
            c_main = Getdesig(imlist[0])
            desig_pix_t0_l.append(c_main.coords.astype(np.int64))
            c_main = Getdesig(imlist[-1])
            goal_pix_l.append(c_main.coords.astype(np.int64))

        images = np.stack(imlist).astype(np.float32) / 255.
        image_batch.append(images)

    image_batch = np.stack(image_batch)

    if desig_pix_t0_l != []:
        desig_pix_t0 = np.stack(desig_pix_t0_l)
        goal_pix = np.stack(goal_pix_l)

    if save_points:
        pkl.dump({'desig_pix_t0': desig_pix_t0, 'goal_pix': goal_pix},
                 open(bench_dir + '/points{}.pkl'.format(grasp_data_mode), 'wb'))

    return image_batch, desig_pix_t0, goal_pi


def annotate_image_vec(images, ann):
    imlist = []
    for b in range(images.shape[0]):
        imlist.append(draw_text_onimage('%.2f' % ann[b], images[b], color=(0,0,255)))
    return np.stack(imlist, 0)


def visuallize_sawyer_track(testdata, conffile, grasp_data_mode, tsteps=120, interval=12):
    hyperparams = imp.load_source('hyperparams', conffile)

    conf = hyperparams.configuration
    images, pix_t0_b, goal_pix_b = load_data(testdata, tsteps, grasp_data_mode=grasp_data_mode, interval=interval)

    modeldata_dir = '/'.join(str.split(conffile, '/')[:-1]) + '/modeldata'
    conf['pretrained_model'] = modeldata_dir + '/model48002'
    conf['batch_size'] = images.shape[0]
    compute_metric(conf, goal_pix_b, images, modeldata_dir, pix_t0_b)


def run_tracking_benchmark():

    load_benchmark_data()



def compute_metric(conf, goal_pix_b, images, modeldata_dir, pix_t0_b, true_desig_pix=None):

    goal_image_warper = setup_gdn(conf, gpu_id=0)
    start_image_b = images[:, 0]
    goal_image_b = images[:, -1]
    bsize = images.shape[0]
    columns = []
    for t in range(images.shape[1]):
        column = []
        warped_image_start, flow_field, start_warp_pts = goal_image_warper(images[:, t], start_image_b)
        warped_image_goal, flow_field, goal_warp_pts = goal_image_warper(images[:, t], goal_image_b)

        curr_im_l = []

        warped_image_start_l = []
        warped_image_goal_l = []
        for b in range(bsize):
            pix_t0 = pix_t0_b[b]
            goal_pix = goal_pix_b[b]

            current_frame = images[b, t]
            warperrs = []
            desig_l = []
            start_image = images[b, 0]
            goal_image = images[b, -1]

            desig_l.append(np.flip(start_warp_pts[b, pix_t0[0], pix_t0[1]], 0))
            start_warperr = np.linalg.norm(start_image[pix_t0[0], pix_t0[1]] -
                                           warped_image_start[b, pix_t0[0], pix_t0[1]])
            warperrs.append(start_warperr)

            desig_l.append(np.flip(goal_warp_pts[b, goal_pix[0], goal_pix[1]], 0))
            goal_warperr = np.linalg.norm(goal_image[goal_pix[0], goal_pix[1]] -
                                          warped_image_goal[b, goal_pix[0], goal_pix[1]])
            warperrs.append(goal_warperr)

            warperrs = np.array([start_warperr, goal_warperr])
            tradeoff = 1 / warperrs / np.sum(1 / warperrs)

            ann_curr = add_crosshairs_single(current_frame, desig_l[0], np.array([1., 0, 0.]))
            ann_curr = add_crosshairs_single(ann_curr, desig_l[1], np.array([0., 0, 1]))
            curr_im_l.append(ann_curr)

            warped_im_start = draw_text_onimage('%.2f' % tradeoff[0], warped_image_start[b], color=(255, 0, 0))
            warped_im_start = add_crosshairs_single(warped_im_start, pix_t0, color=np.array([1., 0, 0.]))
            warped_image_start_l.append(warped_im_start)

            warped_im_goal = draw_text_onimage('%.2f' % tradeoff[1], warped_image_goal[b], color=(255, 0, 0))
            warped_im_goal = add_crosshairs_single(warped_im_goal, goal_pix, color=np.array([0, 0, 1.]))
            warped_image_goal_l.append(warped_im_goal)

        column.append(np.stack(curr_im_l, 0))
        column.append(np.stack(warped_image_start_l, 0))
        column.append(np.stack(warped_image_goal_l, 0))

        newcolumn = []
        for b in range(images.shape[0]):
            for el in column:
                newcolumn.append(el[b])
        column = np.concatenate(newcolumn, 0)
        columns.append(column)
    image = Image.fromarray((np.concatenate(columns, 1) * 255).astype(np.uint8))
    print('imagefile saved to ', modeldata_dir + '/warpstartgoal.png')
    image.save(modeldata_dir + '/warpstartgoal.png')


if __name__ == '__main__':
    # testdata_path = '/mnt/sda1/pushing_data/goaldistancenet_test'
    # conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/weiss_thresh0.5_56x64/conf.py'
    # conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/tdac_weiss_cons0_56x64/conf.py'

    # grasping
    # view = 1
    # testdata_path = '/mnt/sda1/pushing_data/goaldistancenet_test/grasp_2view'
    # # conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/multiview/view{}/conf.py'.format(view)
    # conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/sawyer_grasping_tresh0.5_48x64/conf.py'
    # tsteps = 15
    # interval = 1

    # visuallize_sawyer_track(testdata_path, conffile, grasp_data_mode=view, tsteps=tsteps, interval=interval)

    run_tracking_benchmark()



