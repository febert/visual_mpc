import numpy as np
from matplotlib import animation
import matplotlib.gridspec as gridspec

import imageio
import os
import pdb
import pickle

from matplotlib import pyplot as plt

from python_visual_mpc.video_prediction.misc.makegifs2 import assemble_gif, npy_to_gif
frame = None
canvas = None
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
from python_visual_mpc.utils.txt_in_image import draw_text_image

from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import color_code_distrib
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import compute_overlay


class CEM_Visualizer(object):
    def __init__(self, dict_=None, append_masks=True, filepath=None, dict_name=None, numex = 4, suf= "", col_titles = None, renorm_heatmaps=True, logger=None):
        """
        :param dict_: dictionary containing image tensors
        :param append_masks: whether to visualize the masks
        :param gif_savepath: the path to save the gif
        :param numex: how many examples of the batch to visualize
        :param suf: append a suffix to the gif name
        :param col_titles: a list of titles for each column

        The dictionary contains keys-values pairs of {"video_name":"image_tensor_list"}
        where "video_name" is used as the caption in the visualization
        where "image_tensor_list" is a list with np.arrays (batchsize, 64,64,n_channel) for each time step.

        If n_channel is 1 a heatmap will be shown. Use renorm_heatmaps=True to normalize the heatmaps
        at every time step (this is necessary when the range of values changes significantly over time).

        If the key contains the string "flow" a color-coded flow field will be shown.

        if the key contains the string "masks" the image_tensor_list needs to be of the following form:
        [mask_list_0, ..., mask_list_Tmax]
        where mask_list_t = [mask_0, ..., mask_N]
        where mask_i.shape = [batch_size, 64,64,1]
        """
        print("Starting CEM Visualizer")
        if logger == None:
            self.logger = Logger(mute=True)
        else:
            self.logger = logger

        self.gif_savepath = filepath
        if dict_name != None:
            dict_ = pickle.load(open(filepath + '/' + dict_name, "rb"))

        self.dict_ = dict_

        if 'iternum' in dict_:
            self.iternum = dict_['iternum']
            del dict_['iternum']
        else: self.iternum = ""

        if 'gen_images' in dict_:
            gen_images = dict_['gen_images']
            if gen_images[0].shape[0] < numex:
                raise ValueError("batchsize too small for providing desired number of exmaples!")

        self.numex = numex
        self.video_list = []
        self.append_masks = False

        for key in list(dict_.keys()): #TODO: simplify this!!
            data = dict_[key]
            if 'overlay' in key:
                self.logger.log('visualizing overlay')
                images = data[0]
                gen_distrib = data[1]
                gen_distrib = color_code_distrib(gen_distrib, self.numex, renormalize=True)
                if gen_distrib[0].shape != images[0].shape:
                    images = resize_image(images, gen_distrib[0].shape[1:3])
                overlay = compute_overlay(images, gen_distrib, self.numex)
                self.video_list.append((overlay, key))
            else:
                if isinstance(data, list):
                    if len(data[0].shape) == 4:
                        self.video_list.append((data, key))
                    else:
                        raise "wrong shape in key {} with shape {}".format(key, data[0].shape)
                else:
                    self.logger.log('ignoring key ',key)

            if key == 'scores':
                self.video_list.append((self.get_score_images(data), key))

        self.renormalize_heatmaps = renorm_heatmaps
        self.logger.log('renormalizing heatmaps: ', self.renormalize_heatmaps)

        self.t = 0

        self.suf = suf
        self.num_rows = len(self.video_list)

        self.col_titles = col_titles

    def get_score_images(self, scores):
        height = self.video_list[0][0][0].shape[1]
        width = self.video_list[0][0][0].shape[2]
        seqlen = len(self.video_list[0][0])

        txt_im = []
        for i in range(self.numex):
            txt_im.append(draw_text_image(str(scores[i]), image_size=(height, width)))
        textrow = np.stack(txt_im, 0)

        textrow = [textrow for _ in range(seqlen)]
        return textrow

    def make_direct_vid(self, separate_vid = False, resize=None):
        self.logger.log('making gif with tags')

        new_videolist = []
        for vid in self.video_list:
            # print('key', vid[1])
            # print('len', len(vid[0]))
            # print('sizes', [im.shape for im in vid[0]])
            # print('####')
            # if 'gen_distrib' in vid[1]:
            #     plt.switch_backend('TkAgg')
                # plt.imshow(vid[0][0][0])
                # plt.show()

            images = vid[0]
            if resize is not None:
                images = resize_image(images, size=resize)
            name = vid[1]

            if images[0].shape[-1] == 1 or len(images[0].shape) == 3:
                images = color_code_distrib(images, self.numex, renormalize=True)

            new_videolist.append((images, name))

        framelist = assemble_gif(new_videolist, convert_from_float=True, num_exp=self.numex)
        # save_video_mp4(self.gif_savepath +'/prediction_at_t{}')
        npy_to_gif(framelist, self.gif_savepath +'/direct{}{}'.format(self.iternum,self.suf))

def convert_to_videolist(input, repeat_last_dim):
    tsteps = len(input)
    nmasks = len(input[0])

    list_of_videos = []

    for m in range(nmasks):  # for timesteps
        video = []
        for t in range(tsteps):
            if repeat_last_dim:
                single_mask_batch = np.repeat(input[t][m], 3, axis=3)
            else:
                single_mask_batch = input[t][m]
            video.append(single_mask_batch)
        list_of_videos.append(video)

    return list_of_videos