import numpy as np
from matplotlib import animation
import matplotlib.gridspec as gridspec

import imageio
import os
import pdb
import pickle

from matplotlib import pyplot as plt

from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
frame = None
canvas = None
from python_visual_mpc.utils.txt_in_image import draw_text_image

from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import color_code_distrib
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import compute_overlay


def assemble_gif(video_batch, num_exp = 8, convert_from_float = True, only_ind=None):
    """
    :param video_batch: accepts either
        a list of different video batches
        each video batch is a list of [batchsize, 64, 64, 3] with length timesteps, with type float32 and range 0 to 1
        or each element of the list is tuple (video batch, name)
    or
        a list of tuples with (video_batch, name)

    :param only_ind, only assemble this index
    :return:
    """

    if isinstance(video_batch[0], tuple):
        names = [v[1] for v in video_batch]
        video_batch = [v[0] for v in video_batch]
        txt_im = []
        for name in names:
            txt_im.append(draw_text_image(name, image_size=(video_batch[0][0].shape[1], 200)))
        legend_col = np.concatenate(txt_im, 0)
    else:
        legend_col = None
    vid_length = video_batch[0].shape[1]

    #videobatch is a list of [b, t, r, c, 3]

    fullframe_list = []
    for t in range(vid_length):
        if only_ind is not None:
            column_images = [video[only_ind, t] for video in video_batch]
            full_frame = np.concatenate(column_images, axis=0)  # make column
        else:
            column_list = []
            if legend_col is not None:
                column_list.append(legend_col)

            for exp in range(num_exp):
                column_images = []
                for video in video_batch:
                    column_images.append(video[exp, t])
                column_images = np.concatenate(column_images, axis=0)  #make column
                column_list.append(column_images)

            full_frame = np.concatenate(column_list, axis= 1)

        if convert_from_float:
            full_frame = np.uint8(255 * full_frame)

        fullframe_list.append(full_frame)

    return fullframe_list

def get_score_images(scores, height, width, seqlen, numex):
    txt_im = []
    for i in range(numex):
        txt_im.append(draw_text_image(str(scores[i]), image_size=(height, width)))
    textrow = np.stack(txt_im, 0)
    return np.repeat(textrow[:,None], seqlen, axis=1)

def make_direct_vid(dict, numex, gif_savepath, suf):
    """
    :param dict:  dictionary with video tensors of shape bsize, tlen, r, c, 3
    :param numex:
    :param gif_savepath:
    :param suf:
    :param resize:
    :return:
    """
    new_videolist = []
    shapes = []
    for key in dict:
        images = dict[key]
        print('key', key)
        print('shape', images.shape)

        if len(shapes) > 0:   # check that all the same size
            assert images.shape == shapes[-1], 'shape is different!'
        shapes.append(images.shape)
        assert not isinstance(images, list)

        # if 'gen_distrib' in vid[1]:
        #     plt.switch_backend('TkAgg')
        #     plt.imshow(vid[0][0][0])
        #     plt.show()

        if images[0].shape[-1] == 1 or len(images[0].shape) == 3:
            images = color_code_distrib(images, numex, renormalize=True)
        new_videolist.append((images, key))

    framelist = assemble_gif(new_videolist, convert_from_float=True, num_exp=numex)
    framelist.append(np.zeros_like(framelist[0]))
    # save_video_mp4(gif_savepath +'/prediction_at_t{}')
    npy_to_gif(framelist, gif_savepath +'/direct_{}'.format(suf))
