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
            list_of_maskvideos = get_masks(conf, file_path)
            list_of_maskvideos = [make_color_scheme(v) for v in list_of_maskvideos]
            fused_gif = assemble_gif([ground_truth, gen_images] + list_of_maskvideos, numexp)
        else:
            fused_gif = assemble_gif([ground_truth, gen_images], numexp)

    else:
        gen_images_main = [img[:, :, :, :3] for img in gen_images]
        gen_images_aux1 = [img[:, :, :, 3:] for img in gen_images]
        ground_truth = np.split(ground_truth,ground_truth.shape[1], 1)
        ground_truth = [np.squeeze(img) for img in ground_truth]
        ground_truth = ground_truth[1:]
        ground_truth_main = [img[:, :, :, :3] for img in ground_truth]
        ground_truth_aux1 = [img[:, :, :, 3:] for img in ground_truth]

        fused_gif = assemble_gif([ground_truth_main, gen_images_main, ground_truth_aux1, gen_images_aux1], num_exp= numexp)

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    if not suffix:
        name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
    else: name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    npy_to_gif(fused_gif, name)


def get_masks(conf, file_path, repeat_last_dim = False):
    masks = cPickle.load(open(file_path + '/gen_masks.pkl', "rb"))

    # tsteps = len(masks)
    # nmasks = len(masks[0])
    # print mask statistics:
    # pix_pos = np.array([8, 49])
    # print 'evaluate mask values at designated pixel:',pix_pos
    # for t in range(tsteps):
    #     for imask in range(nmasks):
    #         print 't{0}: mask {1}: value= {2}'.format(t, imask, masks[t][imask][0, pix_pos[0], pix_pos[1]])
    # print 'mask statistics...'
    # for t in range(tsteps):
    #     sum_permask = []
    #     print 'mask of time {}'.format(t)
    #     for imask in range(nmasks):
    #         sum_permask.append(np.sum(masks[t][imask]))
    #         print 'sum of mask{0} :{1}'.format(imask,sum_permask[imask])
    #
    #     sum_all_move = np.sum(np.stack(sum_permask[2:]))
    #     print 'sum of all movment-masks:', sum_all_move
    # end mask statistics:

    return convert_to_videolist(masks, repeat_last_dim)

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


def create_video_pixdistrib_gif(file_path, conf, t=0, suffix = "", n_exp = 8, suppress_number = False,
                                append_masks = False, show_moved= False):
    gen_images = cPickle.load(open(file_path + '/gen_image_t{}.pkl'.format(t), "rb"))


    if  suppress_number:
        name = file_path + '/vid_' + conf['experiment_name'] + suffix
    else:
        itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
        if not suffix:
            name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis)
        else:
            name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    if 'ndesig' in conf:
        gen_distrib1 = cPickle.load(open(file_path + '/gen_distrib1_t{}.pkl'.format(t), "rb"))
        gen_distrib2 = cPickle.load(open(file_path + '/gen_distrib2_t{}.pkl'.format(t), "rb"))

        plot_psum_overtime(conf, gen_distrib1, n_exp, name+'_1', file_path)
        plot_psum_overtime(conf, gen_distrib2, n_exp, name+'_2', file_path)
    else:
        gen_distrib = cPickle.load(open(file_path + '/gen_distrib_t{}.pkl'.format(t), "rb"))
        plot_psum_overtime(conf, gen_distrib, n_exp, name, file_path)

    # trafos = cPickle.load(open(file_path + '/trafos.pkl'.format(t), "rb"))

    makecolor = True

    if 'ndesig' in conf:
        if makecolor:
            gen_distrib1 = make_color_scheme(gen_distrib1)
            gen_distrib2 = make_color_scheme(gen_distrib2)
        else:
            gen_distrib1 = [np.repeat(g, 3, axis=3) for g in gen_distrib1]
            gen_distrib2 = [np.repeat(g, 3, axis=3) for g in gen_distrib2]

        video_list = [gen_images, gen_distrib1, gen_distrib2]
    else:
        if makecolor:
            gen_distrib = make_color_scheme(gen_distrib)
        else:
            gen_distrib = [np.repeat(g, 3, axis=3) for g in gen_distrib]

        video_list = [gen_images, gen_distrib]
    if append_masks:
        list_of_maskvideos = get_masks(conf, file_path, repeat_last_dim=True)
        # list_of_maskvideos = [make_color_scheme(v) for v in list_of_maskvideos]
        video_list += list_of_maskvideos

    if show_moved:
        moved_im = cPickle.load(open(file_path + '/moved_im.pkl', "rb"))
        moved_pix = cPickle.load(open(file_path + '/moved_pix.pkl', "rb"))
        moved_im = convert_to_videolist(moved_im, repeat_last_dim=False)
        moved_pix = convert_to_videolist(moved_pix, repeat_last_dim=True)

        video_list += moved_im
        video_list += moved_pix

    fused_gif = assemble_gif(video_list, n_exp)

    npy_to_gif(fused_gif, name)

def create_video_gif(file_path, conf, t, suffix = None, n_exp = 8):
    gen_images = cPickle.load(open(file_path + '/gen_image_t{}.pkl'.format(t), "rb"))
    name = file_path + '/vid_' + conf['experiment_name'] + suffix
    fused_gif = assemble_gif([gen_images], n_exp)
    npy_to_gif(fused_gif, name)


def plot_psum_overtime(conf, gen_distrib, n_exp, name, filepath):
    plt.figure(figsize=(25, 2),dpi=80)

    if 'avoid_occlusions' in conf:
        occlusioncost = cPickle.load(open(filepath + '/occulsioncost_bestactions.pkl','rb'))

    for ex in range(n_exp):
        psum = []
        ax = plt.subplot(1,n_exp, ex+1)
        for t in range(len(gen_distrib)):
            psum.append(np.sum(gen_distrib[t][ex]))
        psum = np.array(psum)

        if 'avoid_occlusions' in conf:
            ax.set_title("occlusioncost: {}".format(occlusioncost[ex]))

        plt.plot(range(len(gen_distrib)), psum)
        plt.ylim([0,2.5])

    # plt.show()
    plt.savefig(name +"_psum.png")
    plt.close('all')


def go_through_timesteps(file_path):
    for t in range(1,15):
        create_video_pixdistrib_gif(file_path, conf, t, suffix='_t{}'.format(t), n_exp=10, suppress_number=True)


if __name__ == '__main__':
    # file_path = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg'
    file_path = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/dna_multobj'

    # file_path = '/home/frederik/Documents/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd'
    hyperparams = imp.load_source('hyperparams', file_path + '/conf.py')

    conf = hyperparams.configuration
    # conf['visualize'] = conf['output_dir'] + '/model22002'
    # create_video_pixdistrib_gif(file_path, conf, t=1, suppress_number=True)
    # create_video_pixdistrib_gif(exp_dir + '/modeldata', conf, t=0, suppress_number=True, append_masks=True, show_moved=True)
    # create_video_pixdistrib_gif(file_path, conf, n_exp= 10, suppress_number= True)
    #
    go_through_timesteps(file_path +'/verbose')
