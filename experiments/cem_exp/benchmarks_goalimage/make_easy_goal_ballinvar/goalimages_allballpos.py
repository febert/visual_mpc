from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *

from PIL import Image
import cPickle
import random

import pdb
import os
import lsdc

BASE_DIR = '/'.join(str.split(lsdc.__file__, '/')[:-3])
filename = BASE_DIR + '/mjc_models/pushing2d_controller_nomarkers.xml'


def imagespace_to_mujoco(pixel_coord, numpix = 64, noflip =False):
    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    coords = (pixel_coord - float(numpix)/2)/float(numpix) * window_height
    if not noflip:
        mujoco_coords = np.array([-coords[1], coords[0]])
    else:
        mujoco_coords = coords
    return mujoco_coords


def mujoco_to_imagespace(mujoco_coord, numpix = 64, truncate = False, noflip= False):
    """
    convert form Mujoco-Coord to numpix x numpix image space:
    :param numpix: number of pixels of square image
    :param mujoco_coord:
    :return: pixel_coord
    """
    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    pixelwidth = pixelheight
    window_width = pixelwidth * numpix
    middle_pixel = numpix / 2

    if not noflip:
        pixel_coord = np.rint(np.array([-mujoco_coord[1], mujoco_coord[0]]) /
                              pixelwidth + np.array([middle_pixel, middle_pixel]))
    else:
        pixel_coord = np.rint(np.array([mujoco_coord[0], mujoco_coord[1]]) /
                              pixelwidth + np.array([middle_pixel, middle_pixel]))

    pixel_coord = pixel_coord.astype(int)
    if truncate:
        if np.any(pixel_coord < 0) or np.any(pixel_coord > numpix -1):
            print '###################'
            print 'designated pixel is outside the field!! Resetting it to be inside...'
            print 'truncating...'
            if np.any(pixel_coord < 0):
                pixel_coord[pixel_coord < 0] = 0
            if np.any(pixel_coord > numpix-1):
                pixel_coord[pixel_coord > numpix-1]  = numpix-1

    return pixel_coord


def mujoco_get_frame(ballpos, pose, viewer, model):


    # set initial conditions
    model.data.qpos = np.concatenate([ballpos, pose.squeeze(), np.array([0., 0.])])
    model.data.qvel = np.zeros_like(model.data.qvel)

    for _ in range(4):
        model.step()
        viewer.loop_once()

    img_string, width, height = viewer.get_image()
    image = np.fromstring(img_string, dtype='uint8').reshape((64, 64, 3))[::-1, :, :]

    # Image.fromarray(image).show()
    image = image.astype(np.float32)/255.

    # import pdb;
    # pdb.set_trace()
    # viewer.finish()

    return image

def generate_images():

    model = mujoco_py.MjModel(filename)
    viewer = mujoco_py.MjViewer(visible=True, init_width=64, init_height=64)
    viewer.start()
    viewer.cam.camid = 0
    viewer.set_model(model)
    viewer.cam.camid = 0



    nseed = 3
    for i_conf in range(20):
        for seed in range(nseed):
            i_traj = seed + i_conf*nseed
            dict = cPickle.load(open('goalimage/goalimg{0}_conf{1}.pkl'.format(i_traj, i_conf), "rb"))
            block_pose = dict['goal_object_pose']

            d = 4
            num =64/d
            # frames = np.zeros((num, num, 64, 64, 3))
            frames, ballpos = [], []
            for r in range(num):
                for c in range(num):


                    mjc_pos = imagespace_to_mujoco(np.array([r*d, c*d]))
                    print 'run', i_traj, ' r',r*d, 'c',c*d
                    single_frame = mujoco_get_frame(mjc_pos, block_pose, viewer, model)
                    frames.append(single_frame)
                    ballpos.append(np.concatenate([mjc_pos,np.zeros(2)]))

                    # single_frame = single_frame*255.
                    # Image.fromarray(single_frame.astype(np.uint8)).show()
                    print 'ballpos', mjc_pos

                    # print np.sum(model.data.cfrc_ext)
                    # if np.sum(model.data.cfrc_ext) == -0.983262569927:
                    #     single_frame = single_frame*255.
                    #     Image.fromarray(single_frame.astype(np.uint8)).show()

            out_dict = {}
            frames = np.stack(frames)
            ballpos = np.stack(ballpos)
            out_dict['goal_image'] = frames
            out_dict['goal_ballpos'] = ballpos
            cPickle.dump(out_dict, open('goalimage_var_ballpos/goalimg{0}_conf{1}.pkl'.format(i_traj, i_conf), 'wb'))

if __name__ == '__main__':
    random.seed(3)
    np.random.seed(3)
    generate_images()















