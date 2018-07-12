import numpy as np
import os
import collections
import pickle
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs
import pdb


from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import CEM_Visualizer

from python_visual_mpc.utils.txt_in_image import draw_text_onimage

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image

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
    # self.num_exp = I0_t_reals[0].shape[0]
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
    if num_rows== 1:
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
    plt.switch_backend('Agg')
    plt.savefig(os.path.join(dir, filename))
    plt.close()


def write_tradeoff_onimage(image, tradeoff_percam, ntask, startgoal):
    """
    :param tradeoff_percam:
    :param ntask:
    :param startgoal:  0 or 1; 0 stands for startimage
    :return:
    """
    tradeoff_percam = tradeoff_percam.reshape([ntask,2])
    string = ','.join(['%.2f' %  tr for  tr in list(tradeoff_percam[:,startgoal])])
    return draw_text_onimage(string, image)

class CEM_Visual_Preparation(object):
    def __init__(self):
        pass

    def visualize(self, ctrl, actions, scores, cem_itr, gen_distrib, gen_images, last_frames):

        bestindices = scores.argsort()[:ctrl.K]

        print('in make_cem_visuals')
        plt.switch_backend('agg')

        if 'compare_mj_planner_actions' in ctrl.agentparams:
            selindices = np.concatenate([np.zeros(1, dtype=np.int) ,bestindices])
        else: selindices = bestindices
        gen_distrib = gen_distrib[selindices]
        gen_images = gen_images[selindices]
        print('selected distributions')
        if 'image_medium' in ctrl.agentparams:
            gen_distrib = resize_image(gen_distrib, ctrl.goal_image.shape[1:3])
            gen_images = resize_image(gen_images, ctrl.goal_image.shape[1:3])
            print('resized images')

        self.num_ex = selindices.shape[0]
        self.len_pred = ctrl.netconf['sequence_length'] - ctrl.netconf['context_frames']

        print('made directories')
        self.t_dict_ = collections.OrderedDict()

        if 'warp_objective' in ctrl.policyparams:
            print('starting warp objective')
            warped_images = image_addgoalpix(self.num_ex, self.len_pred, ctrl.warped_images, ctrl.goal_pix)
            gen_images = images_addwarppix(gen_images, ctrl.goal_warp_pts_l, ctrl.goal_pix, ctrl.agentparams['num_objects'])
            warped_images = np.split(warped_images[selindices], warped_images.shape[1], 1)
            warped_images = list(np.squeeze(warped_images))
            self.t_dict_['warped_im_t{}'.format(ctrl.t)] = warped_images
            print('warp objective done')

        self.annontate_images(ctrl, gen_images, last_frames)

        if 'use_goal_image' not in ctrl.policyparams or 'comb_flow_warp' in ctrl.policyparams or 'register_gtruth' in ctrl.policyparams:
            self.visualize_goal_pixdistrib(ctrl, gen_distrib, selindices)

        for icam in range(ctrl.ncam):
            print('putting cam: {} res into dict'.format(icam))
            self.t_dict_['gen_images_icam{}_t{}'.format(icam, ctrl.t)] = unstack(gen_images[:, :, icam], 1)

        print('itr{} best scores: {}'.format(cem_itr, [scores[selindices[ind]] for ind in range(self.num_ex)]))
        self.t_dict_['scores'] = scores[selindices]
        self.t_dict_['desig_pix'] = ctrl.desig_pix
        self.t_dict_['goal_pix'] = ctrl.goal_pix

        ctrl.dict_.update(self.t_dict_)
        if 'no_instant_gif' not in ctrl.agentparams:
            v = CEM_Visualizer(self.t_dict_, append_masks=False,
                                   filepath=ctrl.agentparams['record'] + '/plan/',
                                   numex=self.num_ex, suf='t{}iter_{}'.format(ctrl.t, cem_itr))
            if 'image_medium' in ctrl.agentparams:
                size = ctrl.agentparams['image_medium']
            else: size = None
            v.make_direct_vid(resize=size)

        make_action_summary(self.num_ex, actions, ctrl.agentparams, selindices, cem_itr, ctrl.netconf['sequence_length'], ctrl.t)

        if 'save_pkl' in ctrl.agentparams:
            dir = ctrl.agentparams['record'] + '/plan'
            if not os.path.exists(dir):
                os.makedirs(dir)
            pickle.dump(self.t_dict_, open(dir + '/pred_t{}iter{}.pkl'.format(ctrl.t, cem_itr), 'wb'))
            print('written files to:', dir)

    def visualize_goal_pixdistrib(self, ctrl, gen_distrib, selindices):
        for icam in range(ctrl.ncam):
            print('handling case for cam: {}'.format(icam))
            for p in range(ctrl.ndesig):
                sel_gen_distrib_p = unstack(gen_distrib[:, :, icam, :, :, p], 1)
                self.t_dict_['gen_distrib_cam{}_p{}'.format(icam, p)] = sel_gen_distrib_p

    def annontate_images(self, ctrl, last_frames):
        for icam in range(ctrl.ncam):
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            self.t_dict_['curr_img_cam{}'.format(icam)] = unstack(current_image.squeeze(), 1)

    def annontate_goalimage_genimage(self):
        gl_im_ann = None
        gen_image_an_l = None
        print('none registered')
        return gen_image_an_l, gl_im_ann


class CEM_Visual_Preparation_Registration(CEM_Visual_Preparation):

    def annontate_images(self, ctrl, gen_images, last_frames):
        for icam in range(ctrl.ncam):
            print('annotating tracks for cam: {}'.format(icam))
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            current_image = annotate_tracks(ctrl, current_image.squeeze(), icam, self.len_pred, self.num_ex)
            self.t_dict_['curr_img_cam{}'.format(icam)] = unstack(current_image.squeeze(), 1)

        self.visualize_registration(ctrl)

    def visualize_goal_pixdistrib(self, ctrl, gen_distrib, selindices):

        for icam in range(ctrl.ncam):
            print('handling case for cam: {}'.format(icam))
            for p in range(ctrl.ndesig):
                sel_gen_distrib_p = unstack(gen_distrib[:, :, icam, :, :, p], 1)
                self.t_dict_['gen_distrib_cam{}_p{}'.format(icam, p)] = sel_gen_distrib_p
                self.t_dict_['gen_dist_goalim_overlay_cam{}_p{}_t{}'.format(icam, p, ctrl.t)] = \
                    (unstack(self.gl_im_ann_per_tsk[p, :, :, icam], 1), sel_gen_distrib_p)

    def visualize_registration(self, ctrl):

        self.t_dict_['desig_pix_t0'] = ctrl.desig_pix_t0

        if 'image_medium' in ctrl.agentparams:
            pix_mult = ctrl.agentparams['image_medium'][0]/ctrl.agentparams['image_height']
        else:
            pix_mult = 1.

        for icam in range(ctrl.ncam):
            print("on cam: {}".format(icam))
            if 'start' in ctrl.policyparams['register_gtruth']:
                print('on start case')
                if 'trade_off_reg' in ctrl.policyparams:
                    warped_img_start_cam = write_tradeoff_onimage(ctrl.warped_image_start[icam].squeeze(), ctrl.reg_tradeoff[icam],
                                                                  ctrl.ntask, 0)
                else:
                    warped_img_start_cam = ctrl.warped_image_start[icam].squeeze()
                self.t_dict_['warp_start_cam{}'.format(icam)] = [
                    np.repeat(np.expand_dims(warped_img_start_cam, axis=0), self.num_ex, axis=0) for _ in
                    range(self.len_pred)]
                print('finished')

            startimages = np.tile(ctrl.start_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(ctrl.ntask):
                print('on task {}'.format(p))
                if 'image_medium' in ctrl.agentparams:
                    desig_pix_t0 = ctrl.desig_pix_t0_med[icam, p][None]
                else:
                    desig_pix_t0 = ctrl.desig_pix_t0[icam, p][None]
                desig_pix_t0 = np.tile(desig_pix_t0, [self.num_ex, self.len_pred, 1])

                startimages = add_crosshairs(startimages, desig_pix_t0)
            self.t_dict_['start_img_cam{}'.format(icam)] = unstack(startimages, 1)

            for p in range(ctrl.ntask):
                if 'goal' in ctrl.policyparams['register_gtruth']:
                    print('on goal case cam: {}'.format(p))
                    if 'trade_off_reg' in ctrl.policyparams:
                        warped_img_goal_cam = write_tradeoff_onimage(ctrl.warped_image_goal[icam].squeeze(), ctrl.reg_tradeoff[icam],
                                                                     ctrl.ntask, 1)
                    else:
                        warped_img_goal_cam = ctrl.warped_image_goal[icam].squeeze()
                    warped_img_goal_cam = [np.repeat(np.expand_dims(warped_img_goal_cam, axis=0), self.num_ex, axis=0) for _ in
                                           range(self.len_pred)]
                    self.t_dict_['warp_goal_cam{}'.format(icam)] = warped_img_goal_cam

        if 'image_medium' in ctrl.agentparams:
            goal_pix = ctrl.goal_pix_med
        else:
            goal_pix = ctrl.goal_pix

        gl_im_shape = [self.num_ex, self.len_pred, ctrl.ncam] + list(ctrl.goal_image.shape[1:])
        pdb.set_trace()
        gl_im_ann = np.zeros(gl_im_shape)  # b, t, n, r, c, 3
        self.gl_im_ann_per_tsk = np.zeros([ctrl.ndesig] + gl_im_shape)  # p, b, t, n, r, c, 3
        for icam in range(ctrl.ncam):
            print('adding goal pixes {}'.format(icam))
            gl_im_ann[:, :, icam] = np.tile(ctrl.goal_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            self.gl_im_ann_per_tsk[:, :, :, icam] = np.tile(ctrl.goal_image[icam][None, None, None],
                                                       [ctrl.ndesig, self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(ctrl.ndesig):
                gl_im_ann[:, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, gl_im_ann[:, :, icam],
                                                         ctrl.goal_pix[icam, p] * pix_mult)
                self.gl_im_ann_per_tsk[p, :, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, self.gl_im_ann_per_tsk[p][:, :, icam],
                                                                    goal_pix[icam, p])
            self.t_dict_['goal_image{}'.format(icam)] = unstack(gl_im_ann[:, :, icam], 1)
        print("done")


def annotate_tracks(ctrl, current_image, icam, len_pred, num_ex):
    ipix = 0
    for p in range(ctrl.ntask):
        if 'start' in ctrl.policyparams['register_gtruth']:
            desig_pix_start = np.tile(ctrl.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
            current_image = add_crosshairs(current_image, desig_pix_start, color=[1., 0., 0])
            ipix += 1
        if 'goal' in ctrl.policyparams['register_gtruth']:
            desig_pix_goal = np.tile(ctrl.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
            current_image = add_crosshairs(current_image, desig_pix_goal, color=[0, 0, 1.])
            ipix += 1
    return current_image


def unstack(arr, dim):
    orig_dim = list(arr.shape)
    listlen = orig_dim[dim]
    orig_dim.pop(dim)
    newdim = orig_dim
    splitted = np.split(arr, listlen, dim)
    return [el.reshape(newdim) for el in splitted]


def make_action_summary(K, actions, agentparams, bestindices, cem_itr, seq_len, tstep):
    with open(agentparams['record'] + '/plan/actions_t{}iter_{}'.format(tstep, cem_itr), 'w') as f:
        f.write('actions, states \n')
        for i in range(K):
            f.write('k{}\n'.format(i))
            for t_ in range(seq_len):
                f.write('t{}  {}\n'.format(t_, actions[bestindices][i, t_]))

def make_state_summary(K, last_states, gen_states, agentparams, bestindices, cem_itr, tstep):
    with open(agentparams['record'] + '/plan/states_t{}iter_{}'.format(tstep, cem_itr), 'w') as f:
        f.write('last states \n')
        for t_ in range(last_states.shape[1]):
            f.write('t{}  {}\n'.format(t_, last_states[0, t_]))

        f.write('gen states \n')
        for i in range(K):
            f.write('k{}\n'.format(i))
            for t_ in range(gen_states.shape[1]):
                f.write('t{}  {}\n'.format(t_, gen_states[bestindices][i, t_]))
