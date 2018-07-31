import numpy as np
import os
import collections
import pickle
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs
import pdb

from python_visual_mpc.utils.txt_in_image import draw_text_onimage

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import get_score_images

from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import make_direct_vid

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

    def visualize(self, vd):
        """
        :param vd:  visualization data
        :return:
        """

        bestindices = vd.scores.argsort()[:vd.K]
        self.ncam = vd.netconf['ncam']
        self.ndesig = vd.netconf['ndesig']
        self.agentparams = vd.agentparams
        self.policyparams = vd.policyparams

        print('in make_cem_visuals')
        plt.switch_backend('agg')

        if 'compare_mj_planner_actions' in vd.agentparams:
            selindices = np.concatenate([np.zeros(1, dtype=np.int) ,bestindices])
        else: selindices = bestindices
        gen_distrib = vd.gen_distrib[selindices]
        gen_images = vd.gen_images[selindices]
        print('selected distributions')
        if 'image_medium' in vd.agentparams:
            gen_distrib = resize_image(gen_distrib, vd.goal_image.shape[1:3])
            gen_images = resize_image(gen_images, vd.goal_image.shape[1:3])
            print('resized images')

        self.num_ex = selindices.shape[0]
        self.len_pred = vd.netconf['sequence_length'] - vd.netconf['context_frames']

        print('made directories')
        self._t_dict = collections.OrderedDict()

        if 'warp_objective' in vd.policyparams:
            print('starting warp objective')
            warped_images = image_addgoalpix(self.num_ex, self.len_pred, vd.warped_images, vd.goal_pix)
            gen_images = images_addwarppix(gen_images, vd.goal_warp_pts_l, vd.goal_pix, vd.agentparams['num_objects'])
            warped_images = np.split(warped_images[selindices], warped_images.shape[1], 1)
            warped_images = list(np.squeeze(warped_images))
            self._t_dict['warped_im_t{}'.format(vd.t)] = warped_images
            print('warp objective done')

        self.annontate_images(vd, vd.last_frames)

        if 'use_goal_image' not in vd.policyparams or 'comb_flow_warp' in vd.policyparams or 'register_gtruth' in vd.policyparams:
            self.visualize_goal_pixdistrib(vd, gen_distrib)

        for icam in range(self.ncam):
            print('putting cam: {} res into dict'.format(icam))
            self._t_dict['gen_images_icam{}_t{}'.format(icam, vd.t)] = gen_images[:, :, icam]

        print('itr{} best scores: {}'.format(vd.cem_itr, [vd.scores[selindices[ind]] for ind in range(self.num_ex)]))
        self._t_dict['scores'] = get_score_images(vd.scores[selindices], vd.last_frames.shape[3], vd.last_frames.shape[4], self.len_pred, self.num_ex)

        if 'no_instant_gif' not in vd.agentparams:
            if 'image_medium' in vd.agentparams:
                size = vd.agentparams['image_medium']
            else: size = None
            make_direct_vid(self._t_dict, self.num_ex, vd.agentparams['record'] + '/plan/',
                                suf='t{}iter{}'.format(vd.t, vd.cem_itr), resize=size)

        # make_action_summary(self.num_ex, actions, agentparams, selindices, cem_itr, netconf['sequence_length'], t)

        if 'save_pkl' in vd.agentparams:
            dir = vd.agentparams['record'] + '/plan'
            if not os.path.exists(dir):
                os.makedirs(dir)
            pickle.dump(self._t_dict, open(dir + '/pred_t{}iter{}.pkl'.format(vd.t, vd.cem_itr), 'wb'))
            print('written files to:', dir)

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                gen_distrib_ann = color_code_distrib(gen_distrib[:, :, icam, :, :, p], self.num_ex, renormalize=True)
                gen_distrib_ann = image_addgoalpix(self.num_ex, self.len_pred, gen_distrib_ann,
                                                   vd.goal_pix[icam, p])
                self._t_dict['gen_distrib_cam{}_p{}'.format(icam, p)] = gen_distrib_ann

    def annontate_images(self, vd, last_frames):
        for icam in range(self.ncam):
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            self._t_dict['curr_img_cam{}'.format(icam)] = current_image.squeeze()

    def annontate_goalimage_genimage(self):
        gl_im_ann = None
        gen_image_an_l = None
        print('none registered')
        return gen_image_an_l, gl_im_ann


class CEM_Visual_Preparation_Registration(CEM_Visual_Preparation):
    def annontate_images(self, vd, last_frames):
        for icam in range(self.ncam):
            print('annotating tracks for cam: {}'.format(icam))
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            current_image = annotate_tracks(vd, current_image.squeeze(), icam, self.len_pred, self.num_ex)
            self._t_dict['curr_img_cam{}'.format(icam)] = current_image.squeeze()

        self.visualize_registration(vd)

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                sel_gen_distrib_p = gen_distrib[:, :, icam, :, :, p]
                self._t_dict['gen_distrib_cam{}_p{}'.format(icam, p)] = sel_gen_distrib_p
                self._t_dict['gen_dist_goalim_overlay_cam{}_p{}_t{}'.format(icam, p, vd.t)] = \
                compute_overlay(self.gl_im_ann_per_tsk[p, :, :, icam], sel_gen_distrib_p, self.num_ex)

    def visualize_registration(self, vd):
        if 'image_medium' in self.agentparams:
            pix_mult = self.agentparams['image_medium'][0]/self.agentparams['image_height']
        else:
            pix_mult = 1.

        for icam in range(self.ncam):
            print("on cam: {}".format(icam))
            if 'start' in self.policyparams['register_gtruth']:
                print('on start case')
                if 'trade_off_reg' in self.policyparams:
                    warped_img_start_cam = write_tradeoff_onimage(vd.warped_image_start[icam].squeeze(), vd.reg_tradeoff[icam],
                                                                  vd.ntask, 0)
                else:
                    warped_img_start_cam = vd.warped_image_start[icam].squeeze()
                self._t_dict['warp_start_cam{}'.format(icam)] = np.repeat(np.repeat(warped_img_start_cam[None], self.len_pred, axis=0)[None], self.num_ex, axis=0)

            startimages = np.tile(vd.start_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(vd.ntask):
                print('on task {}'.format(p))
                if 'image_medium' in vd.agentparams:
                    desig_pix_t0 = vd.desig_pix_t0_med[icam, p][None]
                else:
                    desig_pix_t0 = vd.desig_pix_t0[icam, p][None]
                desig_pix_t0 = np.tile(desig_pix_t0, [self.num_ex, self.len_pred, 1])

                startimages = add_crosshairs(startimages, desig_pix_t0)
            self._t_dict['start_img_cam{}'.format(icam)] = startimages

            for p in range(vd.ntask):
                if 'goal' in vd.policyparams['register_gtruth']:
                    print('on goal case cam: {}'.format(p))
                    if 'trade_off_reg' in vd.policyparams:
                        warped_img_goal_cam = write_tradeoff_onimage(vd.warped_image_goal[icam].squeeze(), vd.reg_tradeoff[icam], vd.ntask, 1)
                    else:
                        warped_img_goal_cam = vd.warped_image_goal[icam].squeeze()
                    self._t_dict['warp_goal_cam{}'.format(icam)] = np.repeat(np.repeat(warped_img_goal_cam[None], self.len_pred, axis=0)[None], self.num_ex, axis=0)

        if 'image_medium' in vd.agentparams:
            goal_pix = vd.goal_pix_med
        else:
            goal_pix = vd.goal_pix

        gl_im_shape = [self.num_ex, self.len_pred, vd.ncam] + list(vd.goal_image.shape[1:])
        gl_im_ann = np.zeros(gl_im_shape)  # b, t, n, r, c, 3
        self.gl_im_ann_per_tsk = np.zeros([vd.ndesig] + gl_im_shape)  # p, b, t, n, r, c, 3
        for icam in range(vd.ncam):
            print('adding goal pixes {}'.format(icam))
            gl_im_ann[:, :, icam] = np.tile(vd.goal_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            self.gl_im_ann_per_tsk[:, :, :, icam] = np.tile(vd.goal_image[icam][None, None, None],
                                                       [vd.ndesig, self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(vd.ndesig):
                gl_im_ann[:, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, gl_im_ann[:, :, icam],
                                                         goal_pix[icam, p] * pix_mult)
                self.gl_im_ann_per_tsk[p, :, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, self.gl_im_ann_per_tsk[p][:, :, icam],
                                                                    goal_pix[icam, p])
            self._t_dict['goal_image{}'.format(icam)] = gl_im_ann[:, :, icam]


def annotate_tracks(vd, current_image, icam, len_pred, num_ex):
    ipix = 0
    for p in range(vd.ntask):
        if 'start' in vd.policyparams['register_gtruth']:
            desig_pix_start = np.tile(vd.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
            current_image = add_crosshairs(current_image, desig_pix_start, color=[1., 0., 0])
            ipix += 1
        if 'goal' in vd.policyparams['register_gtruth']:
            desig_pix_goal = np.tile(vd.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
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


def compute_overlay(images, distrib, numex):
    color_coded_dist = color_code_distrib(distrib, numex, renormalize=True)
    alpha = .6
    return color_coded_dist*alpha + (1-alpha)*images


def color_code_distrib(inp_distrib, num_ex, renormalize=False):
    out_distrib = []
    for t in range(inp_distrib.shape[1]):
        distrib = inp_distrib[:,t]
        out_t = []

        for b in range(num_ex):
            cmap = plt.cm.get_cmap('jet')
            if renormalize:
                distrib[b] /= (np.max(distrib[b])+1e-6)
            colored_distrib = cmap(np.squeeze(distrib[b]))[:, :, :3]
            out_t.append(colored_distrib)

        out_t = np.stack(out_t, 0)
        out_distrib.append(out_t)
    return np.stack(out_distrib, 1)
