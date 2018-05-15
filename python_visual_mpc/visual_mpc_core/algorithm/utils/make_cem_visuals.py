import numpy as np
import pdb
import os
import collections
from PIL import Image
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import cv2
import pdb

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def image_addgoalpix(bsize, seqlen, image_l, goal_pix):
    goal_pix_ob = np.tile(goal_pix[None, None, :], [bsize, seqlen, 1])
    return add_crosshairs(image_l, goal_pix_ob)

def images_addwarppix(gen_images, warp_pts_l, pix, num_objects):
    warp_pts_arr = np.stack(warp_pts_l, axis=1)
    for ob in range(num_objects):
        warp_pts_ob = warp_pts_arr[:, :, pix[ob, 0], pix[ob, 1]]
        gen_images = add_crosshairs(gen_images, np.flip(warp_pts_ob, 2))
    return gen_images

def plot_sum_overtime(pixdistrib, dir, filename, tradeoffs):

    # shape pixdistrib: b, t, icam, r, c, ndesig
    # num_exp = I0_t_reals[0].shape[0]
    b, seqlen, ncam, r, c, ndesig = pixdistrib.shape
    num_rows = ncam*ndesig
    num_cols = b

    pixdistrib = np.sum(pixdistrib, 3)
    pixdistrib = np.sum(pixdistrib, 3)

    print('num_rows', num_rows)
    print('num_cols', num_cols)
    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    if tradeoffs is not None:
        num_rows += 1

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    print('start')
    row = 0
    if ncam == 1:
        icam = 0
        for col in range(num_cols):
            for p in range(ndesig):
                axarr[col].plot(range(seqlen), pixdistrib[col,:,icam,p])
                axarr[col].set_ylim([0, 3])
    else:
        for col in range(num_cols):
            for icam in range(ncam):
                for p in range(ndesig):
                    row = icam*ndesig + p
                    axarr[row, col].plot(range(seqlen), pixdistrib[col,:,icam,p])
                    axarr[row, col].set_ylim([0, 3])

    if tradeoffs is not None:
        for p in range(ndesig):
            row += 1
            for col in range(num_cols):
                axarr[row, col].plot(range(seqlen), tradeoffs[col, :, 0, p])  # plot the first value of tradeoff
                axarr[row, col].set_ylim([0, 3])

    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, filename))


def make_cem_visuals(ctrl, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images, gen_states,
                     last_frames, goal_warp_pts_l, scores, warped_image_goal, warped_image_start, warped_images):

        bestindices = bestindices[:ctrl.K]
        len_pred = ctrl.netconf['sequence_length'] - ctrl.netconf['context_frames']
        bsize = ctrl.netconf['batch_size']

        if ctrl.save_subdir != None:
            file_path = ctrl.policyparams['current_dir'] + '/' + ctrl.save_subdir + '/verbose'
        else:
            file_path = ctrl.policyparams['current_dir'] + '/verbose'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        t_dict_ = collections.OrderedDict()

        for icam in range(ctrl.ncam):
            current_image = [np.repeat(np.expand_dims(last_frames[0, 1, icam], axis=0), ctrl.K, axis=0) for _ in
                             range(len_pred)]
            t_dict_['current_image_cam{}'.format(icam)] = current_image

        if 'warp_objective' in ctrl.policyparams:
            warped_images = image_addgoalpix(bsize, len_pred, warped_images, ctrl.goal_pix)
            gen_images = images_addwarppix(gen_images, goal_warp_pts_l, ctrl.goal_pix, ctrl.agentparams['num_objects'])
            warped_images = np.split(warped_images[bestindices], warped_images.shape[1], 1)
            warped_images = list(np.squeeze(warped_images))
            t_dict_['warped_im_t{}'.format(ctrl.t)] = warped_images

        if 'register_gtruth' in ctrl.policyparams:

            gen_image_an_l = []
            for icam in range(ctrl.ncam):
                if 'start' in ctrl.policyparams['register_gtruth']:
                    t_dict_['warped_image_start_cam{}'.format(icam)] = [np.repeat(np.expand_dims(warped_image_start.squeeze(), axis=0), ctrl.K, axis=0) for _ in
                                                      range(len_pred)]

                startimages = np.tile(ctrl.start_image[None], [ctrl.K, len_pred, 1, 1, 1])
                if 'image_medium' in ctrl.agentparams:
                    desig_pix_t0 = ctrl.desig_pix_t0_med[icam][None]
                else:
                    desig_pix_t0 = ctrl.desig_pix_t0[icam][None]
                desig_pix_t0 = np.tile(desig_pix_t0, [ctrl.K, len_pred, 1])
                t_dict_['start_image_cam{}'.format(icam)] = add_crosshairs(startimages, desig_pix_t0)

                ipix = 0
                gen_image_an = gen_images[:, :, icam]
                if 'start' in ctrl.policyparams['register_gtruth']:
                    desig_pix_start = np.tile(ctrl.desig_pix[icam, 0][None, None, :], [bsize, len_pred, 1])
                    gen_image_an = add_crosshairs(gen_image_an, desig_pix_start, color=[1., 0., 0])
                    ipix +=1
                if 'goal' in ctrl.policyparams['register_gtruth']:
                    desig_pix_goal = np.tile(ctrl.desig_pix[icam, ipix][None,None, :], [bsize, len_pred, 1])
                    gen_image_an = add_crosshairs(gen_image_an, desig_pix_goal, color=[0, 0, 1.])
                gen_image_an_l.append(gen_image_an)


                t_dict_['warped_image_goal_cam{}'.format(icam)] = [
                    np.repeat(np.expand_dims(warped_image_goal.squeeze(), axis=0), ctrl.K, axis=0) for _ in
                    range(len_pred)]
        gen_image_an_l = None

        goal_image_annotated = []
        for icam in range(ctrl.ncam):
            goal_image = [np.repeat(np.expand_dims(ctrl.goal_image[icam], axis=0), ctrl.K, axis=0) for _ in
                          range(len_pred)]
            for p in range(ctrl.ndesig):
                goal_image_annotated.append(image_addgoalpix(ctrl.K , len_pred, goal_image, ctrl.goal_pix[icam, p]))
            t_dict_['goal_image{}'.format(icam)] = goal_image_annotated[icam]

        if 'use_goal_image' not in ctrl.policyparams or 'comb_flow_warp' in ctrl.policyparams or 'register_gtruth' in ctrl.policyparams:
            sel_gen_distrib = gen_distrib[bestindices]
            if hasattr(ctrl, 'tradeoffs'):
                tradeoffs = ctrl.tradeoffs[bestindices]
            else: tradeoffs = None
            plot_sum_overtime(sel_gen_distrib, ctrl.agentparams['record'] + '/plan', 'psum_t{}_iter{}'.format(ctrl.t, cem_itr), tradeoffs)
            for icam in range(ctrl.ncam):
                for p in range(ctrl.ndesig):
                    sel_gen_distrib_p = unstack(sel_gen_distrib[:,:, icam,:,:, p], 1)
                    t_dict_['gen_distrib_cam{}_p{}_t{}'.format(icam, p, ctrl.t)] = sel_gen_distrib_p
                    t_dict_['gen_distrib_goalim_overlay_cam{}_{}_t{}'.format(icam, p, ctrl.t)] = (goal_image_annotated[icam], sel_gen_distrib_p)

        for icam in range(ctrl.ncam):
            if gen_image_an_l is not None:
                t_dict_['gen_images_icam{}_t{}'.format(icam, ctrl.t)] = unstack(gen_image_an_l[icam][bestindices], 1)
            else:
                t_dict_['gen_images_icam{}_t{}'.format(icam, ctrl.t)] = unstack(gen_images[bestindices,:,icam], 1)

        print('itr{} best scores: {}'.format(cem_itr, [scores[bestindices[ind]] for ind in range(ctrl.K)]))
        ctrl.dict_.update(t_dict_)
        if 'no_instant_gif' not in ctrl.agentparams:
            v = Visualizer_tkinter(t_dict_, append_masks=False,
                                   filepath=ctrl.agentparams['record'] + '/plan/',
                                   numex=ctrl.K, suf='t{}iter_{}'.format(ctrl.t, cem_itr))
            if 'image_medium' in ctrl.agentparams:
                size = ctrl.agentparams['image_medium']
            else: size = None
            v.make_direct_vid(resize=size)

            start_frame_conc = np.concatenate([last_frames[0, 0, 0], last_frames[0, 1, 0]], 0).squeeze()
            start_frame_conc = (start_frame_conc*255.).astype(np.uint8)
            Image.fromarray(start_frame_conc).save(ctrl.agentparams['record'] + '/plan/start_frame{}iter_{}.png'.format(ctrl.t, cem_itr))

            make_state_action_summary(ctrl.K, actions, ctrl.agentparams, bestindices, cem_itr, gen_states, ctrl.netconf['sequence_length'], ctrl.t)
            if 'warp_objective' in ctrl.policyparams:
                t_dict_['warp_pts_t{}'.format(ctrl.t)] = sel_func(goal_warp_pts_l)
                t_dict_['flow_fields{}'.format(ctrl.t)] = flow_fields[bestindices[:K]]


def unstack(arr, dim):
    orig_dim = list(arr.shape)
    listlen = orig_dim[dim]
    orig_dim.pop(dim)
    newdim = orig_dim
    splitted = np.split(arr, listlen, dim)
    return [el.reshape(newdim) for el in splitted]

def make_state_action_summary(K, actions, agentparams, bestindices, cem_itr, gen_states, seqlen, tstep):
    with open(agentparams['record'] + '/plan/actions_states_t{}iter_{}'.format(tstep, cem_itr), 'w') as f:
        f.write('actions, states \n')
        for i in range(K):
            f.write('k{}\n'.format(i))
            for t_ in range(15):
                f.write('t{}  {}\n'.format(t_, actions[bestindices][i, t_]))
