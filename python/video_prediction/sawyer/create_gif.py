import cPickle
from video_prediction.utils_vpred.create_gif import *
import numpy as np
import imp
import re
import pdb

def create_gif(file_path, conf, suffix = None, numexp = 8, append_masks = False):
    print 'reading files from:', file_path
    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image.pkl', "rb"))


    ground_truth = np.squeeze(ground_truth)
    if ground_truth.shape[4] == 3:

        ground_truth = np.split(ground_truth,ground_truth.shape[1], 1)
        ground_truth = [np.squeeze(img) for img in ground_truth]
        ground_truth = ground_truth[1:]

        if append_masks:
            list_of_maskvideos = get_masks(file_path)
            list_of_maskvideos = [make_color_scheme(v) for v in list_of_maskvideos]
            fused_gif = assemble_gif([ground_truth, gen_images] + list_of_maskvideos, numexp)
        else:
            fused_gif = assemble_gif([ground_truth, gen_images], numexp)

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


def get_masks(file_path):
    masks = cPickle.load(open(file_path + '/gen_masks.pkl', "rb"))

    tsteps = len(masks)
    nmasks = len(masks[0])
    list_of_maskvideos = []

    for m in range(nmasks):  # for timesteps
        mask_video = []
        for t in range(tsteps):
            # single_mask_batch = np.repeat(masks[t][m], 3, axis=3 )
            single_mask_batch = masks[t][m]
            mask_video.append(single_mask_batch)
        list_of_maskvideos.append(mask_video)

    return list_of_maskvideos


def create_video_pixdistrib_gif(file_path, conf, t, suffix = "", n_exp = 8, suppress_number = False):
    gen_images = cPickle.load(open(file_path + '/gen_image_t{}.pkl'.format(t), "rb"))
    gen_distrib = cPickle.load(open(file_path + '/gen_distrib_t{}.pkl'.format(t), "rb"))

    if  suppress_number:
        name = file_path + '/vid_' + conf['experiment_name'] + suffix
    else:
        itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
        if not suffix:
            name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
        else:
            name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    if 'single_view' in conf:
        plot_psum_overtime(gen_distrib, n_exp, name)

    if 'single_view' not in conf:
        gen_images_main = [img[:, :, :, :3] for img in gen_images]
        gen_images_aux1 = [img[:, :, :, 3:] for img in gen_images]

        gen_distrib_main = [d[:, :, :, 0] for d in gen_distrib]
        gen_distrib_aux1 = [d[:, :, :, 1] for d in gen_distrib]

        gen_distrib_main = make_color_scheme(gen_distrib_main)
        gen_distrib_aux1 = make_color_scheme(gen_distrib_aux1)

        fused_gif = assemble_gif([gen_images_main, gen_distrib_main, gen_images_aux1, gen_distrib_aux1], n_exp)
    else:
        makecolor = True
        if makecolor:
            gen_distrib = make_color_scheme(gen_distrib)
        else:
            gen_distrib = [np.repeat(g, 3, axis=3) for g in gen_distrib]

        fused_gif = assemble_gif([gen_images, gen_distrib], n_exp)

    npy_to_gif(fused_gif, name)

def create_video_gif(file_path, conf, t, suffix = None, n_exp = 8):
    gen_images = cPickle.load(open(file_path + '/gen_image_t{}.pkl'.format(t), "rb"))
    name = file_path + '/vid_' + conf['experiment_name'] + suffix
    fused_gif = assemble_gif([gen_images], n_exp)
    npy_to_gif(fused_gif, name)


def plot_psum_overtime(gen_distrib, n_exp, name):
    plt.figure(figsize=(25, 2),dpi=80)

    for ex in range(n_exp):
        psum = []
        plt.subplot(1,n_exp, ex+1)
        for t in range(len(gen_distrib)):
            psum.append(np.sum(gen_distrib[t][ex]))

        psum = np.array(psum)
        plt.plot(range(len(gen_distrib)), psum)
        plt.ylim([0,2.5])

    # plt.show()
    plt.savefig(name +"_psum.png")
    plt.close('all')


def go_through_timesteps(filepath):
    for t in range(1,9):
        create_video_pixdistrib_gif(file_path, conf, t, suffix='_t{}'.format(t), n_exp=10, suppress_number=True)


if __name__ == '__main__':
    file_path = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/avoid_occlusions_firstplan/verbose'
    hyperparams = imp.load_source('hyperparams', '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/avoid_occlusions_firstplan/conf.py')
    # file_path = '/home/guser/Desktop/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop/verbose'
    # hyperparams = imp.load_source('hyperparams', '/home/guser/Desktop/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop/conf.py')
    conf = hyperparams.configuration
    # conf['visualize'] = conf['output_dir'] + '/model22002'
    create_video_pixdistrib_gif(file_path, conf, t=1, suppress_number=True)
    # create_video_pixdistrib_gif(file_path, conf, n_exp= 10, suppress_number= True)
    #
    # go_through_timesteps(file_path)
