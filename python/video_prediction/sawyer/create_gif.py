import cPickle
from video_prediction.utils_vpred.create_gif import *
import numpy as np
import imp
import re
import pdb

def create_gif(file_path, conf, suffix = None, numexp = 8):
    print 'reading files from:', file_path
    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image.pkl', "rb"))

    ground_truth = np.squeeze(ground_truth)
    if ground_truth.shape[4] == 3:

        ground_truth = np.split(ground_truth,ground_truth.shape[1], 1)
        ground_truth = [np.squeeze(img) for img in ground_truth]
        ground_truth = ground_truth[1:]

        fused_gif = assemble_gif([ground_truth, gen_images])

    else:
        gen_images_main = [img[:, :, :, :3] for img in gen_images]
        gen_images_aux1 = [img[:, :, :, 3:] for img in gen_images]
        ground_truth = np.split(ground_truth,ground_truth.shape[1], 1)
        ground_truth = [np.squeeze(img) for img in ground_truth]
        ground_truth_main = [img[:, :, :, :3] for img in ground_truth]
        ground_truth_aux1 = [img[:, :, :, 3:] for img in ground_truth]

        fused_gif = assemble_gif([ground_truth_main, gen_images_main, ground_truth_aux1, gen_images_aux1], num_exp= numexp)

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    if not suffix:
        name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
    else: name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    npy_to_gif(fused_gif, name)

def create_video_pixdistrib_gif(file_path, conf, suffix = None, n_exp = 8):
    gen_images = cPickle.load(open(file_path + '/gen_image.pkl', "rb"))
    gen_distrib = cPickle.load(open(file_path + '/gen_distrib.pkl', "rb"))

    gen_distrib = make_color_scheme(gen_distrib)

    fused_gif = assemble_gif([gen_images, gen_distrib], n_exp)

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    if not suffix:
        name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
    else:
        name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    npy_to_gif(fused_gif, name)




if __name__ == '__main__':
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/sawyer/singleview_shifted/modeldata'
    hyperparams = imp.load_source('hyperparams', '/home/frederik/Documents/lsdc/tensorflow_data/sawyer/singleview_shifted/conf.py')
    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model114002'
    pred = create_video_pixdistrib_gif(file_path, conf, suffix='diffmotions')