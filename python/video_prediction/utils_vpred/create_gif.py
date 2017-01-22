import glob
import moviepy.editor as mpy
import numpy as np
import cPickle
import os
import imp
from PIL import Image
import re
import matplotlib.pyplot as plt


def npy_to_gif(im_list, filename):
    clip = mpy.ImageSequenceClip(im_list, fps=4)
    clip.write_gif(filename + '.gif')
    return

def comp_video(file_path, conf, suffix = None):
    print 'reading files from:', file_path
    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image_seq.pkl', "rb"))

    ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
    ground_truth = np.squeeze(ground_truth)

    fused_gif = assemble_gif([ground_truth, gen_images])

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)

    if not suffix:
        name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
    else: name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix
    npy_to_gif(fused_gif, name)

    return fused_gif

def comp_single_video(file_path, ground_truth):
    ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
    ground_truth = np.squeeze(ground_truth)

    fused_gif = assemble_gif([ground_truth])
    npy_to_gif(fused_gif, file_path)

def pix_distrib_video(pix_distrib):
    """
    change to jet colorscheme, mark maximum value pixel
    :return:
    """
    new_pix_distrib_list = []

    for t in range(len(pix_distrib)):


        new_pix_distrib = np.zeros((pix_distrib[0].shape[0], 64, 64, 3), dtype=np.float32)

        for b in range(pix_distrib[0].shape[0]):

            img = pix_distrib[t][b].squeeze()

            fig = plt.figure(figsize=(1, 1), dpi=64)
            fig.add_subplot(111)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            axes = plt.gca()
            plt.cla()

            axes.axis('off')
            plt.imshow(img, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
            axes.autoscale(False)

            fig.canvas.draw()  # draw the canvas, cache the renderer

            # plt.show()

            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            data = data.astype(np.float32) / 256.0
            new_pix_distrib[b] = data

            # import pdb;
            # pdb.set_trace()
            # plt.show()
            # Image.fromarray(np.uint8(data*255)).show()

        new_pix_distrib_list.append(new_pix_distrib)

        # pdb.set_trace()
    return new_pix_distrib_list

def add_crosshairs(distrib, pix_list):
    """
    add crosshairs to video
    :param distrib:
    :param pix_list: list of x, y coords
    :return:
    """
    for i in range(len(distrib)):
        x, y = pix_list[i]
        distrib[i][0, x] = 0
        distrib[i][0, :, y] = 0

    return distrib


def comp_pix_distrib(file_path, name= None, masks = False, examples = 8):
    pix_distrib = cPickle.load(open(file_path + '/gen_distrib.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_images.pkl', "rb"))
    gtruth_images = cPickle.load(open(file_path + '/gtruth_images.pkl', "rb"))

    print 'finished loading'

    pix_distrib = pix_distrib_video(pix_distrib)

    videolist = [gtruth_images, gen_images, pix_distrib]

    suffix = ''
    if masks:
        gen_masks = cPickle.load(open(file_path + '/gen_masks.pkl', "rb"))
        mask_videolist = []
        nummasks = len(gen_masks[0])
        tsteps = len(gen_masks)
        for m in range(nummasks):
            mask_video = []
            for t in range(tsteps):
                 mask_video.append(np.repeat(gen_masks[t][m], 3, axis=3))

            mask_videolist.append(mask_video)
        videolist += mask_videolist
        suffix = "_masks"

    # exp = 5
    # for i in range(14):
    #     pic = pix_distrib[i].squeeze()[exp]
    #     img = Image.fromarray(np.uint8(255. * pic), mode='L')
    #     img.show()
    #     print i, 'sum:  ', np.sum(pic)
    #     print i, 'max: ', np.max(pic)
    #     pic = np.sort(pic.flatten())
    #     print '10 largest values : ', pic[-10:]

    fused_gif = assemble_gif(videolist, num_exp= examples)
    if not name:
        npy_to_gif(fused_gif, file_path + '/gen_images_pix_distrib'+ suffix)
    else:
        npy_to_gif(fused_gif, file_path + '/' + name + suffix)


def assemble_gif(video_batch, num_exp = 8):
    """
    accepts a list of different video batches
    each video batch is a list of [batchsize, 64, 64, 3] with length timesteps
    :param image_rows:
    :return:
    """

    vid_length = min([len(vid) for vid in video_batch])
    print 'video length:', vid_length
    for i in range(len(video_batch)):
        video_batch[i] = [np.expand_dims(videoframe, axis=0) for videoframe in video_batch[i]]

    # import pdb; pdb.set_trace()
    for i in range(len(video_batch)):
        video_batch[i] = np.concatenate(video_batch[i], axis= 0)

    #videobatch is a list of [timelength, batchsize, 64, 64, 3]

    fullframe_list = []
    for t in range(vid_length):
        column_list = []
        for exp in range(num_exp):
            column_images = [video[t,exp] for video in video_batch]
            column_images = np.concatenate(column_images, axis=0)  #make column
            column_list.append(column_images)

        full_frame = np.concatenate(column_list, axis= 1)
        full_frame = np.uint8(255 * full_frame)

        fullframe_list.append(full_frame)

    return fullframe_list

def comp_masks(file_path, conf, pred = None, suffix = None):
    masks = cPickle.load(open(file_path + '/mask_list.pkl', "rb"))
    mask_list = []

    num_exp = 8  #one colmun per example
    tsteps = len(masks)
    nmasks = len(masks[0])

    for i in range(tsteps): # for timesteps
        row_list = []
        for k in range(nmasks):  # for batch instances (number of rows)
            mask_comp = [np.uint8(255 * masks[i][k][j]) for j in range(num_exp)]
            row_list.append(np.concatenate(tuple(mask_comp), axis=1))

        stacked_rows = np.concatenate(tuple(row_list), axis=0)

        if pred != None:
            pred_rows = pred[i]
            stacked_rows = np.repeat(stacked_rows,3,axis=2)
            stacked_rows = np.concatenate((stacked_rows, pred_rows), axis=0)

        mask_list.append(stacked_rows)

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    if not suffix:
        name = file_path + '/masks_' + conf['experiment_name'] + '_' + str(itr_vis)
    else: name = file_path + '/masks_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix
    npy_to_gif(mask_list, name)

if __name__ == '__main__':

    # splitted = str.split(os.path.dirname(__file__), '/')
    # file_path = '/'.join(splitted[:-3] + ['tensorflow_data/skip_frame/use_every4'])
    # file_path = '/home/frederik/Documents/lsdc/tensorflow_data/skip_frame/use_every_4'

    # file_path = '/home/frederik/Documents/lsdc/tensorflow_data/downsized_less_layers/modeldata'
    # hyperparams = imp.load_source('hyperparams', '/home/frederik/Documents/lsdc/tensorflow_data/downsized_less_layers/conf.py' )
    # conf = hyperparams.configuration
    # conf['visualize'] = conf['output_dir'] + '/model98002'
    # pred = comp_video(file_path, conf)

    # file_path = '/home/frederik/Documents/lsdc/experiments/cem_exp/data_files'
    # comp_pix_distrib(file_path, masks= False)

    # file = '/home/frederik/Documents/lsdc/experiments/cem_exp/data_files'

    file = '/home/frederik/Documents/lsdc/experiments/cem_exp/videos_distrib'

    l = cPickle.load(open(file + '/correction.pkl', "rb"))

    orig_images, gen_images, corr_distrib, desig_pix = l[0], l[1], l[2], l[3]



    corr_distrib = pix_distrib_video(corr_distrib)
    corr_distrib = add_crosshairs(corr_distrib, desig_pix)

    orig_images = [img.astype(np.float32) / 255. for img in orig_images]

    frame_list = assemble_gif([orig_images, gen_images, corr_distrib], num_exp= 1)


    npy_to_gif(frame_list, file + '/correction')