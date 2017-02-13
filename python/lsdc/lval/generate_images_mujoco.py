from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *

from PIL import Image
import cPickle
import random

import os
cwd = os.getcwd()

BASE_DIR = '/'.join(str.split(cwd, '/')[:-3])
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
    model.data.qpos = np.concatenate([ballpos, pose])
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

    for i in range(5):

        block_pos = np.random.uniform(-.35, .35, 2)
        alpha = np.random.uniform(0, np.pi * 2)
        ori = np.array([np.cos(alpha / 2), 0, 0, np.sin(alpha / 2)])
        pose = np.concatenate((block_pos, np.array([0]), ori), axis=0)  #format: x, y, z, quat

        print 'placing block at: ', block_pos, 'corresponds to pixel', mujoco_to_imagespace(block_pos, noflip=True)

        desig_pos = block_pos

        d = 1
        num = 64/d
        frames = np.zeros((num, num, 64, 64, 3))
        for r in range(num):
            for c in range(num):

                mjc_pos = imagespace_to_mujoco(np.array([r*d, c*d]))
                print 'run', i, ' r',r*d, 'c',c*d
                single_frame = mujoco_get_frame(mjc_pos, pose, viewer, model)
                frames[r,c] = single_frame

                if (r * 64 + c) % 1 == 0:
                    single_frame = single_frame*255.
                    Image.fromarray(single_frame.astype(np.uint8)).show()

                import pdb; pdb.set_trace()

        # cPickle.dump([desig_pos, frames], open('query_images/im_collection{}.pkl'.format(i), 'wb'))


if __name__ == '__main__':
    random.seed(3)
    np.random.seed(3)
    generate_images()