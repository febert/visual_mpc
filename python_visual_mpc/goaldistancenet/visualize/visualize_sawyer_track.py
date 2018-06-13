from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
import os
import numpy as np
import cv2
import pickle as pkl

from python_visual_mpc.utils.txt_in_image import draw_text_onimage
import sys
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import imp
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.benckmarking.annotate_benchmark import get_folders
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single
from PIL import Image

FLAGS = flags.FLAGS

def load_data(bench_dir, tsteps, num_ex=4):

    # bench_dir = bench_dir
    # folders = get_folders(bench_dir)
    folders = [bench_dir + '/ob18s1']

    image_batch = []
    desig_pix_t0_l = []
    goal_pix_l = []


    for folder in folders:

        exp_dir = folder + '/videos'
        name = str.split(folder, '/')[-1]
        print('example: ', name)

        desig_pix_t0 = pkl.load(open(exp_dir + '/desig_goal_pixstart.pkl', 'rb'), encoding='latin1')['desig_pix'][0]
        goal_pix = pkl.load(open(exp_dir + '/desig_goal_pixgoal.pkl', 'rb'), encoding='latin1')['desig_pix'][0]

        desig_pix_t0_l.append(desig_pix_t0.astype(np.int))
        goal_pix_l.append(goal_pix.astype(np.int))

        imlist = []
        for t in range(0, tsteps, 12):
            imlist.append(np.array(Image.open(exp_dir + '/small{}.jpg'.format(t))))

        images = np.stack(imlist).astype(np.float32)/255.
        image_batch.append(images)

    image_batch = np.stack(image_batch)
    desig_pix_t0 = np.stack(desig_pix_t0_l)
    goal_pix = np.stack(goal_pix_l)

    return image_batch, desig_pix_t0, goal_pix

def annotate_image_vec(images, ann):
    imlist = []
    for b in range(images.shape[0]):
        imlist.append(draw_text_onimage('%.2f' % ann[b], images[b], color=(0,0,255)))
    return np.stack(imlist, 0)


def visuallize_sawyer_track(testdata, conffile):
    hyperparams = imp.load_source('hyperparams', conffile)

    conf = hyperparams.configuration
    tsteps = 120
    images, pix_t0_b, goal_pix_b = load_data(testdata, tsteps)

    modeldata_dir = '/'.join(str.split(conffile, '/')[:-1]) + '/modeldata'
    conf['pretrained_model'] = modeldata_dir + '/model48002'
    conf['batch_size'] = images.shape[0]
    goal_image_warper = setup_gdn(conf, gpu_id=0)

    start_image_b = images[:,0]
    goal_image_b = images[:,-1]



    bsize = images.shape[0]
    columns = []

    for t in range(images.shape[1]):
        column = []
        warped_image_start, flow_field, start_warp_pts = goal_image_warper(images[:,t], start_image_b)
        warped_image_goal, flow_field, goal_warp_pts = goal_image_warper(images[:,t], goal_image_b)

        curr_im_l = []

        warped_image_start_l = []
        warped_image_goal_l = []
        for b in range(bsize):

            pix_t0 = pix_t0_b[b]
            goal_pix = goal_pix_b[b]

            current_frame = images[b,t]
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
            tradeoff = 1/warperrs/np.sum(1/warperrs)

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

    image = Image.fromarray((np.concatenate(columns, 1)*255).astype(np.uint8))
    image.save(modeldata_dir + '/warpstartgoal.png')

if __name__ == '__main__':
    testdata_path = '/mnt/sda1/pushing_data/goaldistancenet_test'
    conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/weiss_thresh0.5_56x64/conf.py'
    # conffile = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/tdac_weiss_cons0_56x64/conf.py'

    visuallize_sawyer_track(testdata_path, conffile)



