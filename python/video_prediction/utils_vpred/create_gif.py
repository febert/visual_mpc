import glob
import moviepy.editor as mpy
import numpy as np
import cPickle
import os
import imp
from PIL import Image
import re


def npy_to_gif(im_list, filename):
    clip = mpy.ImageSequenceClip(im_list, fps=10)
    clip.write_gif(filename + '.gif')
    return

def comp_video(file_path, conf):
    print 'reading files from:', file_path
    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image_seq.pkl', "rb"))
    collected_pairs = []
    num_exp =8
    for j in range(num_exp):

        ground_truth_list = list(np.uint8(255*ground_truth[j]))

        gen_image_list =[]
        for i in range(len(gen_images)):
            gen_image_list.append(np.uint8(255*gen_images[i][j]))

        column_list = [np.concatenate((truth, gen), axis=0)
                       for truth, gen in zip(ground_truth_list, gen_image_list)]

        collected_pairs.append(column_list)

    fused_gif = []
    for i in range(len(collected_pairs[0])):
        frame_list = [collected_pairs[j][i] for j in range(num_exp)]
        fused_gif.append(np.concatenate( tuple(frame_list), axis= 1))


    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    npy_to_gif(fused_gif, file_path +'/' + conf['experiment_name'] + '_' + str(itr_vis))

    return fused_gif

def comp_masks(file_path, conf, pred = None):
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
    npy_to_gif(mask_list, file_path + '/masks_' + conf['experiment_name'] + '_' + str(itr_vis))

if __name__ == '__main__':

    # splitted = str.split(os.path.dirname(__file__), '/')
    # file_path = '/'.join(splitted[:-3] + ['tensorflow_data/skip_frame/use_every4'])
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/skip_frame/use_every_4'

    hyperparams = imp.load_source('hyperparams', file_path + '/conf.py')
    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model16002'
    pred = comp_video(file_path, conf)


    # masks = comp_masks(file_path, pred)