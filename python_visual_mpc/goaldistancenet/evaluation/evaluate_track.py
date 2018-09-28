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
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single
from PIL import Image
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig

from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif

import glob
import re

import pdb
save_points = False
FLAGS = flags.FLAGS

from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import make_direct_vid
from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import get_score_images


REGION_AVERAGING = True

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_folders(dir):
    files = glob.glob(dir + '/*')
    folders = []
    for f in files:
        if os.path.isdir(f):
            folders.append(f)
    return sorted_alphanumeric(folders)

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

def load_benchmark_data(conf):

    bench_dir = conf['bench_dir']
    view = conf['view']
    if isinstance(bench_dir, list):
        folders = []
        for source_dir in bench_dir:
            folders += get_folders(source_dir)
    else:
        folders = get_folders(bench_dir)

    num_ex = conf['batch_size']
    folders = folders[:num_ex]

    image_batch = []
    desig_pix_t0_l = []
    goal_pix_l = []
    goal_image_l = []
    true_desig_l = []

    for folder in folders:
        name = str.split(folder, '/')[-1]
        print('example: ', name)
        exp_dir = folder + '/images{}'.format(view)

        imlist = []
        for t in range(30):
            imname = exp_dir + '/im_{}.jpg'.format(t)
            im = np.array(Image.open(imname))
            orig_imshape = im.shape
            im = cv2.resize(im, (conf['orig_size'][1], conf['orig_size'][0]), interpolation=cv2.INTER_AREA)
            imlist.append(im)

        images = np.stack(imlist).astype(np.float32) / 255.
        image_batch.append(images)

        image_size_ratio = conf['orig_size'][0]/orig_imshape[0]

        true_desig = np.load(folder + '/points.npy')
        true_desig = (true_desig[view]*image_size_ratio).astype(np.int)
        true_desig_l.append(true_desig)

        goal_image_l.append(images[-1])
        desig_pix_t0_l.append(true_desig[0])
        goal_pix_l.append(true_desig[-1])


    image_batch = np.stack(image_batch)
    desig_pix_t0 = np.stack(desig_pix_t0_l)
    goal_pix = np.stack(goal_pix_l)
    true_desig = np.stack(true_desig_l, axis=0)
    goal_images = np.stack(goal_image_l, axis=0)

    return image_batch, desig_pix_t0, goal_pix, true_desig, goal_images


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


def run_tracking_benchmark(conf):

    image_batch, desig_pix_t0, goal_pix, true_desig, goal_images = load_benchmark_data(conf)

    compute_metric(conf, image_batch, goal_pix, desig_pix_t0, true_desig, goal_images)



def compute_metric(conf, images, goal_pix_b, pix_t0_b, true_desig_pix_b=None, goal_images=None):

    goal_image_warper = setup_gdn(conf, gpu_id=0)
    if goal_images is None:
        goal_image_b = images[:, -1]
    else: goal_image_b = goal_images

    columns = []

    # plt.switch_backend('TkAgg')
    # plt.imshow(start_image_b[0])
    # plt.show()

    bsize, seq_len, im_height, im_width, _ = images.shape
    warped_images_start = np.zeros([bsize, seq_len, im_height, im_width, 3])
    warped_images_goal = np.zeros([bsize, seq_len, im_height, im_width, 3])
    curr_im = np.zeros([bsize, seq_len, im_height, im_width, 3])

    pos_error_start = np.zeros([bsize, seq_len])
    pos_error_goal = np.zeros([bsize, seq_len])
    warperror_start = np.zeros([bsize, seq_len])
    warperror_goal = np.zeros([bsize, seq_len])

    tinterval = 1
    tstart = 0
    start_image_b = images[:, tstart]

    tradeoff_all = np.zeros([bsize, seq_len, 2])

    for t in range(tstart, seq_len, tinterval):
        column = []
        warped_image_start, flow_field, start_warp_pts = goal_image_warper(images[:, t, None], start_image_b[:,None])
        warped_image_goal, flow_field, goal_warp_pts = goal_image_warper(images[:, t, None], goal_image_b[:,None])

        curr_im_l = []

        warped_image_start_l = []
        warped_image_goal_l = []
        for b in range(bsize):
            pix_t0 = pix_t0_b[b]
            goal_pix = goal_pix_b[b]
            true_desig_pix = true_desig_pix_b[b, t]

            current_frame = images[b, t]
            warperrs = []
            desig_l = []
            start_image = images[b, 0]
            goal_image = images[b, -1]

            if REGION_AVERAGING:
                print('using region averaging!!')
                width = 5
                r_range = np.clip(np.array((pix_t0[0]-width,pix_t0[0]+width+1)), 0, im_height-1)
                c_range = np.clip(np.array((pix_t0[1]-width,pix_t0[1]+width+1)), 0, im_width-1)

                point_field = start_warp_pts[b, 0, r_range[0]:r_range[1], c_range[0]:c_range[1]]
                desig_l.append(np.flip(np.array([np.median(point_field[:,:,0]), np.median(point_field[:,:,1])]), axis=0))

                region_tradeoff = True
                if region_tradeoff:
                    start_warperr = np.mean(np.square(start_image[r_range[0]:r_range[1], c_range[0]:c_range[1]] -
                                                      warped_image_start[b, 0, r_range[0]:r_range[1], c_range[0]:c_range[1]]))
                else:
                    start_warperr = np.linalg.norm(start_image[pix_t0[0], pix_t0[1]] -
                                                   warped_image_start[b, 0, pix_t0[0], pix_t0[1]])
                warperrs.append(start_warperr)

                r_range = np.clip(np.array((goal_pix[0]-width,goal_pix[0]+width+1)), 0, im_height)
                c_range = np.clip(np.array((goal_pix[1]-width,goal_pix[1]+width+1)), 0, im_width)

                point_field = goal_warp_pts[b, 0, r_range[0]:r_range[1], c_range[0]:c_range[1]]
                desig_l.append(np.flip(np.array([np.median(point_field[:,:,0]), np.median(point_field[:,:,1])]), axis=0))

                if region_tradeoff:
                    goal_warperr = np.mean(np.square(goal_image[r_range[0]:r_range[1], c_range[0]:c_range[1]] -
                                                      warped_image_goal[b, 0, r_range[0]:r_range[1], c_range[0]:c_range[1]]))
                else:
                    goal_warperr = np.linalg.norm(goal_image[goal_pix[0], goal_pix[1]] -
                                                  warped_image_goal[b, 0, goal_pix[0], goal_pix[1]])
                warperrs.append(goal_warperr)
            else:
                desig_l.append(np.flip(start_warp_pts[b, 0, pix_t0[0], pix_t0[1]], 0))
                start_warperr = np.linalg.norm(start_image[pix_t0[0], pix_t0[1]] -
                                               warped_image_start[b, 0, pix_t0[0], pix_t0[1]])
                warperrs.append(start_warperr)

                desig_l.append(np.flip(goal_warp_pts[b, 0, goal_pix[0], goal_pix[1]], 0))
                goal_warperr = np.linalg.norm(goal_image[goal_pix[0], goal_pix[1]] -
                                              warped_image_goal[b, 0, goal_pix[0], goal_pix[1]])
                warperrs.append(goal_warperr)

            warperror_start[b, t] = start_warperr
            warperror_goal[b, t] = goal_warperr


            warperrs = np.array([start_warperr, goal_warperr])
            tradeoff = 1 / warperrs / np.sum(1 / warperrs)

            tradeoff_all[b,t] = tradeoff

            ann_curr = current_frame
            ann_curr = add_crosshairs_single(ann_curr, desig_l[0], np.array([1., 0, 0.]), thick=True)
            ann_curr = add_crosshairs_single(ann_curr, desig_l[1], np.array([0., 0, 1]), thick=True)
            # ann_curr = add_crosshairs_single(ann_curr, true_desig_pix, np.array([0., 1, 1]), thick=True)
            curr_im_l.append(ann_curr)

            # pos_error_start[b,t] = np.linalg.norm(pix_t0 - true_desig_pix)
            # pos_error_goal[b,t] = np.linalg.norm(goal_pix - true_desig_pix)
            pos_error_start[b,t] = np.linalg.norm(desig_l[0] - true_desig_pix)
            pos_error_goal[b,t] = np.linalg.norm(desig_l[1] - true_desig_pix)

            warped_im_start = draw_text_onimage('%.2f' % tradeoff[0], warped_image_start[b, 0], color=(255, 0, 0))
            warped_im_start = add_crosshairs_single(warped_im_start, pix_t0, color=np.array([1., 0, 0.]), thick=True)
            warped_image_start_l.append(warped_im_start)

            warped_im_goal = draw_text_onimage('%.2f' % tradeoff[1], warped_image_goal[b, 0], color=(255, 0, 0))
            warped_im_goal = add_crosshairs_single(warped_im_goal, goal_pix, color=np.array([0, 0, 1.]), thick=True)
            warped_image_goal_l.append(warped_im_goal)

        column.append(np.stack(curr_im_l, 0))
        column.append(np.stack(warped_image_start_l, 0))
        column.append(np.stack(warped_image_goal_l, 0))

        curr_im[:,t] = np.stack(curr_im_l, 0)
        warped_images_start[:,t] = np.stack(warped_image_start_l, 0)
        warped_images_goal[:,t] = np.stack(warped_image_goal_l, 0)

        newcolumn = []
        numex = 5
        for b in range(numex):
            for el in column:
                newcolumn.append(el[b])

        columns.append(np.concatenate(newcolumn, 0))

    # plot_trackerrors(conf['output_dir'], pos_error_start, pos_error_goal, warperror_start, warperror_goal, tradeoff_all)
    # make_gifs(curr_im, warped_images_start, warped_images_goal, conf)

    image = Image.fromarray((np.concatenate(columns, 1) * 255).astype(np.uint8))
    file = conf['output_dir'] + '/warpstartgoal.png'
    print('imagefile saved to ', file)
    image.save(file)

    write_scores(conf, pos_error_start, pos_error_goal, tradeoff_all)


def plot_trackerrors(outdir, pos_errstart, pos_errgoal, warperr_start, warperr_goal, tradeoff):

    outdir = outdir + '/bench_plots'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for b in range(pos_errstart.shape[0]):
        plt.switch_backend('TkAgg')
        plt.figure()
        plt.plot(pos_errstart[b], label='poserr_start')
        plt.plot(pos_errgoal[b], label='poserr_goal')

        plt.plot(warperr_start[b]*10, '--', label='warperr_start')
        plt.plot(warperr_goal[b]*10, '--', label='warperr_goal')

        plt.plot(tradeoff[b,:,0]*10, label='tradeoff')

        plt.legend()
        # plt.show()
        plt.savefig(outdir + '/traj{}.png'.format(b))


def make_gifs(curr_im, warped_images_start, warped_images_goal, conf):
    bsize = curr_im.shape[0]
    numbers = get_score_images(range(bsize), curr_im.shape[2], curr_im.shape[3], curr_im.shape[1], curr_im.shape[0])

    dict = {'curr_im':curr_im, 'warped_start':warped_images_start, 'warped_goal':warped_images_goal, ' ':numbers}
    make_direct_vid(dict, curr_im.shape[0], conf['output_dir'], suf='')

# def make_gifs(columns, conf):
#     columns = [(col * 255.).astype(np.uint8) for col in columns]
#     columns.append(np.zeros_like(columns[0]))
#     npy_to_gif(columns, conf['output_dir'] + '/warpstartgoal')


def write_scores(conf, pos_error_start, pos_error_goal, tradeoff):
    result_file = conf['output_dir'] + '/pos_error.txt'
    f = open(result_file, 'w')

    avg_minerror_startgoal = np.mean(np.min(np.stack([pos_error_start, pos_error_goal], axis=0), axis=0), axis=1)

    # pos_error_start = np.mean(pos_error_start, axis=1)
    # pos_error_goal = np.mean(pos_error_goal, axis=1)

    f.write('avg distance (over all) {} min per tstep in pixels \n'.format(np.mean(avg_minerror_startgoal)))
    f.write('avg distance (over all) min per tstep {} ratio\n'.format(np.mean(avg_minerror_startgoal)/conf['orig_size'][0]))
    f.write('median distance (over all) {} min per tstep in pixels \n'.format(np.median(avg_minerror_startgoal)))
    f.write('median distance (over all) min per tstep {} ratio\n'.format(np.median(avg_minerror_startgoal)/conf['orig_size'][0]))

    avg_error_startgoal = np.mean(np.mean(np.stack([pos_error_start, pos_error_goal], axis=0), axis=0), axis=1)
    f.write('median distance (over all) avg per tstep {} ratio\n'.format(np.median(avg_error_startgoal)/conf['orig_size'][0]))

    avg_maxerror_startgoal = np.mean(np.max(np.stack([pos_error_start, pos_error_goal], axis=0), axis=0), axis=1)
    f.write('median distance (over all) max per tstep {} ratio\n'.format(np.median(avg_maxerror_startgoal)/conf['orig_size'][0]))

    avg_weighted_error = np.mean(pos_error_start*tradeoff[:,:,0] + pos_error_goal*tradeoff[:,:,1], axis=1)
    f.write('median reg-weighted distance {} ratio\n'.format(np.median(avg_weighted_error)/conf['orig_size'][0]))

    min_reg = np.argmin(tradeoff, axis=2)
    pos_error_startgoal = pos_error_start*min_reg + pos_error_goal*(1-min_reg)
    avg_min_reg_error = np.mean(pos_error_startgoal, axis=1)
    f.write('median hardmin distance {} ratio\n'.format(np.median(avg_min_reg_error)/conf['orig_size'][0]))

    f.write('pos_error start, pos_error goal, avg over min \n')
    for n in range(pos_error_start.shape[0]):
        f.write('{}: {}, {}, {}, {} \n'.format(n, np.mean(pos_error_start[n]), np.mean(pos_error_goal[n]), avg_minerror_startgoal[n], avg_min_reg_error[n]))

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


    view = 0
    # conffile = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/weiss/multiview_multiscale_96x128_highpenal/view{}/conf.py'.format(view)
    # conffile = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/weiss/multiview_new_env_len8/view{}/conf.py'.format(view)
    conffile = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/weiss/multiview_new_env_96x128_len8/view{}/conf.py'.format(view)
    # conffile = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/weiss/smoothcost_only_96x128/conf.py'.format(view)


    hyperparams = imp.load_source('hyperparams', conffile)
    conf = hyperparams.configuration
    modeldata_dir = '/'.join(str.split(conffile, '/')[:-1]) + '/modeldata'
    conf['pretrained_model'] = [modeldata_dir + '/model56002']


    conf['bench_dir'] = ['/mnt/sda1/pushing_data/sawyer_grasping/eval/track_annotations']

    conf['batch_size'] = 20
    run_tracking_benchmark(conf)



