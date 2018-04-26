import numpy as np
import pickle
import colorsys
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.misc

from PIL import Image

def visualize_flow(flow_vecs):
    bsize = flow_vecs[0].shape[0]
    T = len(flow_vecs)

    magnitudes = [np.linalg.norm(f, axis=2) + 1e-5 for f in flow_vecs]
    max_magnitude = np.max(magnitudes)
    norm_magnitudes = [m / max_magnitude for m in magnitudes]

    magnitudes = [np.expand_dims(m, axis=3) for m in magnitudes]

    #pixelflow vectors normalized for unit length
    norm_flow = [np.divide(f, m) for f, m in zip(flow_vecs, magnitudes)]
    flow_angle = [np.arctan2(p[:, :, 0], p[:, :, 1]) for p in norm_flow]
    color_flow = [np.zeros((64, 64, 3)) for _ in range(T)]

    for t in range(T):
        for r in range(64):
            for c in range(64):
                color_flow[t][r, c] = colorsys.hsv_to_rgb((flow_angle[t][r, c] +np.pi) / 2 / np.pi,
                                                          norm_magnitudes[t][r, c],
                                                          1.)
    return color_flow


def rgb2gray(rgb):
    grey = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grey = np.expand_dims(grey, axis=2)
    grey = np.repeat(grey, axis=2, repeats=3)
    # plt.imshow(grey[:,:,1])
    # plt.show()
    return grey


class Image_Creator(object):
    def __init__(self, i_ex, dict_ = None, filepath=None, renorm_heatmaps=True):
        """
        :param dict_: dictionary containing image tensors
        :param append_masks: whether to visualize the masks
        :param gif_savepath: the path to save the gif
        :param i_ex: how many examples of the batch to visualize
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

        if dict_ == None:
            dict_ = pickle.load(open(filepath + '/pred.pkl', "rb"))

        self.image_folder = os.path.join(filepath, dict_['exp_name'])
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        if 'iternum' in dict_:
            self.iternum = dict_['iternum']
        else: self.iternum = 0

        if 'gen_images' in dict_:
            gen_images = dict_['gen_images']
            if gen_images[0].shape[0] < i_ex:
                raise ValueError("batchsize too small for providing desired index of exmaples!")

        self.i_ex = i_ex
        self.video_list = []

        for key in list(dict_.keys()):
            print('processing key {}'.format(key))
            data = dict_[key]

            if key ==  'ground_truth':  # special treatement for gtruth
                ground_truth = dict_['ground_truth']
                if not isinstance(ground_truth, list):
                    ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
                    if ground_truth[0].shape[0] == 1:
                        ground_truth = [g.reshape((1,64,64,3)) for g in ground_truth]
                    else:
                        ground_truth = [np.squeeze(g) for g in ground_truth]

                ground_truth = [im[i_ex] for im in ground_truth]
                self.save_imlist(upsample_nearest(ground_truth[:2]), 'context_frames')
                ground_truth = ground_truth[1:]
                ground_truth = upsample_nearest(ground_truth)
                self.save_imlist(ground_truth, 'images')

            elif 'flow' in key:
                print('visualizing key {} with colorflow'.format(key))

                flow = [im[i_ex] for im in data]
                flow = visualize_flow(flow)
                flow = upsample_nearest(flow)
                self.save_imlist(flow, 'flow')

            elif 'distrib' in key:
                distrib = [d[i_ex] for d in data]
                distrib = color_code_distrib(distrib, renorm_heatmaps)
                self.save_imlist(distrib, 'gen_distrib')

                ground_truth = [rgb2gray(im) for im in ground_truth]
                # plt.imshow(ground_truth[0])
                # plt.show()
                overlaid = compute_overlay(ground_truth, distrib)
                self.save_imlist(overlaid, 'gen_distrib_overlaid')

            elif 'gen_images' in key:  # for a single video channel
                images = [im[i_ex] for im in data]
                images = upsample_nearest(images)
                self.save_imlist(images, 'gen_images')

            elif type(data[0]) is np.ndarray:  # for a single video channel
                images = [im[i_ex] for im in data]
                images = upsample_nearest(images)
                self.save_imlist(images, key)


    def save_imlist(self, imlist, name):

        folder = os.path.join(self.image_folder, name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        for t, im in enumerate(imlist):
            assert im.dtype == np.uint8
            im = Image.fromarray(im)
            im.save(os.path.join(folder, '%05d_%02d.png'%(self.i_ex, t)))

def upsample_nearest(imlist, size = (256,256)):
    out = []
    for im in imlist:
        out.append(scipy.misc.imresize(im, size, 'nearest'))

    return out

def compute_overlay(images, color_coded_distrib):
    alpha = .6
    output_list = []
    for im, distrib in zip(images, color_coded_distrib):

        im = im.astype(np.float32)
        distrib = distrib.astype(np.float32)

        fused = distrib * alpha + (1 - alpha) * im
        output_list.append(fused.astype(np.uint8))
    return output_list


def color_code_distrib(distrib_list, renormalize=False):
    print('renormalizing heatmaps: ', renormalize)
    out_distrib = []
    for distrib in distrib_list:
        cmap = plt.cm.get_cmap('jet')
        if renormalize:
            distrib /= (np.max(distrib)+1e-5)
        distrib = cmap(np.squeeze(distrib))[:, :, :3]

        distrib = scipy.misc.imresize(distrib, [256,256], 'bilinear')

        out_distrib.append(distrib)

    return out_distrib


def main():
    file_path = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/orig_cdna_wristrot_k17_generatescratchimage_bs16/modeldata'

    i_ex = 0
    Image_Creator(i_ex, filepath=file_path, renorm_heatmaps=True)

if __name__ == '__main__':
    main()