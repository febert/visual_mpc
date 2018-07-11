""" This file defines the linear Gaussian policy class. """
import pdb
import numpy as np

import pdb
import os
import copy
import time
import imp
import pickle
from datetime import datetime
import copy
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs

from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import make_cem_visuals
# from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import copy
import pdb
from scipy.special import expit
import collections
import cv2
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
from python_visual_mpc.goaldistancenet.variants.multiview_testgdn import MulltiviewTestGDN
from python_visual_mpc.goaldistancenet.variants.multiview_testgdn import MulltiviewTestGDN
from Queue import Queue
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from threading import Thread
if "NO_ROS" not in os.environ:
    from visual_mpc_rospkg.msg import floatarray
    from rospy.numpy_msg import numpy_msg
    import rospy

import time
from .utils.cem_controller_utils import save_track_pkl, standardize_and_tradeoff, compute_warp_cost, construct_initial_sigma, reuse_cov, reuse_mean, truncate_movement, get_mask_trafo_scores, make_blockdiagonal

from .cem_controller_base import CEM_Controller_Base

verbose_queue = Queue()
def verbose_worker():
    req = 0
    while True:
        print('servicing req', req)
        try:
            plt.switch_backend('Agg')
            ctrl, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images, gen_states, \
            last_frames, goal_warp_pts_l, scores, warped_image_goal, \
            warped_image_start, warped_images, last_states, reg_tradeoff = verbose_queue.get(True)
            make_cem_visuals(ctrl, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images, gen_states,
                             last_frames, goal_warp_pts_l, scores, warped_image_goal,
                             warped_image_start, warped_images, last_states, reg_tradeoff)
        except RuntimeError:
            print("TKINTER ERROR, SKIPPING")
        req += 1

class CEM_Controller_Vidpred(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor=None, goal_image_warper=None, save_subdir=None):
        """
        :param ag_params:
        :param policyparams:
        :param predictor:
        :param save_subdir:
        :param gdnet: goal-distance network
        """
        CEM_Controller_Base.__init__(self, ag_params, policyparams)

        hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
        self.netconf = hyperparams.configuration
        if 'gdnconf' in self.policyparams:
            hyperparams = imp.load_source('hyperparams', self.policyparams['gdnconf'])
            self.gdnconf = hyperparams.configuration
        self.bsize = self.netconf['batch_size']
        self.seqlen = self.netconf['sequence_length']

        assert self.naction_steps * self.repeat == self.seqlen

        self.ncontxt = self.netconf['context_frames']
        self.predictor = predictor
        self.goal_image_warper = goal_image_warper
        self.goal_image = None

        if 'ndesig' in self.netconf:
            self.ndesig = self.netconf['ndesig']
        else: self.ndesig = None
        if 'ntask' in self.agentparams:   # number of
            self.ntask = self.agentparams['ntask']
        else: self.ntask = 1

        self.img_height, self.img_width = self.netconf['orig_size']

        if 'cameras' in self.agentparams:
            self.ncam = len(self.agentparams['cameras'])
        else: self.ncam = 1
        self.rec_input_distrib = []  # record the input distributions

        if 'sawyer' in self.agentparams:
            self.gen_image_publisher = rospy.Publisher('gen_image', numpy_msg(floatarray), queue_size=10)
            self.gen_pix_distrib_publisher = rospy.Publisher('gen_pix_distrib', numpy_msg(floatarray), queue_size=10)
            self.gen_score_publisher = rospy.Publisher('gen_score', numpy_msg(floatarray), queue_size=10)

        self.desig_pix = None
        self.goal_mask = None
        self.goal_pix = None
        if 'trade_off_reg' not in self.policyparams:
            self.reg_tradeoff = np.ones([self.ncam, self.ndesig])/self.ncam/self.ndesig

        if 'cameras' in self.agentparams:
            self.ncam = len(self.agentparams['cameras'])
        else: self.ncam = 1
        self.rec_input_distrib = []  # record the input distributions
        self._thread = Thread(target=verbose_worker)
        self._thread.start()
        self.goal_image = None

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.naction_steps * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_factor
        return actions_costs


    def switch_on_pix(self, desig):
        one_hot_images = np.zeros((1, self.netconf['context_frames'], self.ncam, self.img_height, self.img_width, self.ndesig), dtype=np.float32)
        desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self.img_height, self.img_width]).reshape((1, 2)) - 1).astype(np.int)
        # switch on pixels
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                one_hot_images[:, :, icam, desig[icam, p, 0], desig[icam, p, 1], p] = 1.
                self.logger.log('using desig pix',desig[icam, p, 0], desig[icam, p, 1])
        return one_hot_images

    def get_rollouts(self, traj, actions, cem_itr, itr_times):
        t_0 = time.time()
        ctxt = self.netconf['context_frames']

        last_frames = traj.images[self.t - ctxt + 1:self.t + 1]  # same as [t - 1:t + 1] for context 2
        last_frames = last_frames.astype(np.float32, copy=False) / 255.
        last_frames = last_frames[None]
        if 'register_gtruth' in self.policyparams and cem_itr == 0:
            self.start_image = copy.deepcopy(self.traj.images[0]).astype(np.float32) / 255.
            self.warped_image_start, self.warped_image_goal, self.reg_tradeoff = self.register_gtruth(self.start_image, last_frames, self.goal_image)

        if 'image_medium' in self.agentparams:   # downsample to video-pred reslution
            last_frames = resize_image(last_frames, (self.img_height, self.img_width))

        if 'use_vel' in self.netconf:
            last_states = traj.X_Xdot_full[self.t - ctxt + 1:self.t + 1]
        else:
            last_states = traj.X_full[self.t - ctxt + 1:self.t + 1]

        if 'autograsp' in self.agentparams:
            last_states = last_states[:,:5] #ignore redundant finger dim
            if 'finger_sensors' in self.agentparams:
                touch = traj.touch_sensors[self.t - ctxt + 1:self.t + 1]
                last_states = np.concatenate([last_states, touch], axis=1)
            actions = actions[:,:,:self.netconf['adim']]
        last_states = last_states[None]

        if 'use_goal_image' in self.policyparams and 'comb_flow_warp' not in self.policyparams \
                and 'register_gtruth' not in self.policyparams:
            input_distrib = None
        elif 'masktrafo_obj' in self.policyparams:
            curr_obj_mask = np.repeat(self.curr_obj_mask[None], self.netconf['context_frames'], axis=0).astype(np.float32)
            input_distrib = np.repeat(curr_obj_mask[None], self.M, axis=0)[...,None]
        else:
            input_distrib = self.make_input_distrib(cem_itr)

        self.logger.log('t0 ', time.time() - t_0)
        t_startpred = time.time()

        if 'compare_mj_planner_actions' in self.agentparams:
            actions[0] = self.traj.mj_planner_actions

        if self.M > self.bsize:
            nruns = self.M//self.bsize
            assert self.bsize*nruns == self.M
        else:
            nruns = 1
            assert self.M == self.bsize
        gen_images_l, gen_distrib_l, gen_states_l = [], [], []
        itr_times['pre_run'] = time.time() - t_0
        for run in range(nruns):
            self.logger.log('run{}'.format(run))
            t_run_loop = time.time()
            actions_ = actions[run*self.bsize:(run+1)*self.bsize]
            gen_images, gen_distrib, gen_states, _ = self.predictor(input_images=last_frames,
                                                                    input_state=last_states,
                                                                    input_actions=actions_,
                                                                    input_one_hot_images=input_distrib)
            gen_images_l.append(gen_images)
            gen_distrib_l.append(gen_distrib)
            gen_states_l.append(gen_states)
            itr_times['run{}'.format(run)] = time.time() - t_run_loop
        t_run_post = time.time()
        gen_images = np.concatenate(gen_images_l, 0)
        gen_distrib = np.concatenate(gen_distrib_l, 0)
        if gen_states_l[0] is not None:
            gen_states = np.concatenate(gen_states_l, 0)
        itr_times['t_concat'] = time.time() - t_run_post
        self.logger.log('time for videoprediction {}'.format(time.time() - t_startpred))
        t_run_post = time.time()
        t_startcalcscores = time.time()
        scores_per_task = []
        flow_fields, warped_images, goal_warp_pts_l = None, None, None
        if 'use_goal_image' in self.policyparams and not 'register_gtruth' in self.policyparams:
            # evaluate images with goal-distance network
            goal_image = np.repeat(self.goal_image[None], self.M, axis=0)
            if 'MSE_objective' in self.policyparams:
                scores = self.MSE_based_score(gen_images, goal_image)
            elif 'warp_objective' in self.policyparams:
                flow_fields, scores, goal_warp_pts_l, warped_images = self.flow_based_score(gen_images, goal_image)
            warp_scores = copy.deepcopy(scores)

        if 'masktrafo_obj' in self.policyparams:
            scores = get_mask_trafo_scores(self.policyparams, gen_distrib, self.goal_mask)

        if 'use_goal_image' not in self.policyparams or 'comb_flow_warp' in self.policyparams or 'register_gtruth' in self.policyparams:
            if 'trade_off_views' in self.policyparams:
                scores_per_task, self.tradeoffs = self.compute_trade_off_cost(gen_distrib)
            else:
                for icam in range(self.ncam):
                    for p in range(self.ndesig):
                        distance_grid = self.get_distancegrid(self.goal_pix[icam, p])
                        score = self.calc_scores(icam, p, gen_distrib[:,:, icam, :,:, p], distance_grid, normalize=True)
                        if 'trade_off_reg' in self.policyparams:
                            score *= self.reg_tradeoff[icam, p]
                        scores_per_task.append(score)
                        self.logger.log('best flow score of task {} cam{}  :{}'.format(p, icam, np.min(scores_per_task[-1])))
                scores_per_task = np.stack(scores_per_task, axis=1)

                if 'only_take_first_view' in self.policyparams:
                    scores_per_task = scores_per_task[:,0][:,None]

            scores = np.mean(scores_per_task, axis=1)
            if 'compare_mj_planner_actions' in self.agentparams:
                score_mj_planner_actions = scores[0]
                print('scores for trajectory of mujoco planner',score_mj_planner_actions)
                scores = scores[1:]

            bestind = scores.argsort()[0]
            for icam in range(self.ncam):
                for p in range(self.ndesig):
                    self.logger.log('flow score of best traj for task{} cam{} :{}'.format(p, icam, scores_per_task[bestind, p + icam*self.ndesig]))

            self.best_cost_perstep = self.cost_perstep[bestind]

            # for predictor_propagation only!!
            if 'predictor_propagation' in self.policyparams:
                assert not 'correctorconf' in self.policyparams
                if cem_itr == (self.policyparams['iterations'] - 1):
                    # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                    bestind = scores.argsort()[0]
                    best_gen_distrib = gen_distrib[bestind, self.ncontxt].reshape(1, self.ncam, self.img_height, self.img_width, self.ndesig)
                    self.rec_input_distrib.append(best_gen_distrib)

            flow_scores = copy.deepcopy(scores)

        if 'comb_flow_warp' in self.policyparams:
            scores = standardize_and_tradeoff(flow_scores, warp_scores, self.policyparams['comb_flow_warp'])

        self.logger.log('time to calc scores {}'.format(time.time()-t_startcalcscores))


        bestindices = scores.argsort()[:self.K]
        itr_times['run_post'] = time.time() - t_run_post
        tstart_verbose = time.time()

        if self.verbose and cem_itr == self.policyparams['iterations']-1 and self.i_tr % self.verbose_freq ==0 or \
                ('verbose_every_itr' in self.policyparams and self.i_tr % self.verbose_freq ==0):

            verbose_queue.put((self, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images,
                               gen_states, last_frames, goal_warp_pts_l, scores, self.warped_image_goal,
                               self.warped_image_start, warped_images, last_states, self.reg_tradeoff))

        if 'save_desig_pos' in self.agentparams:
            save_track_pkl(self, self.t, cem_itr)

            # if 'sawyer' in self.agentparams:
            # bestind = self.publish_sawyer(gen_distrib, gen_images, scores)

        if 'store_video_prediction' in self.agentparams and \
                cem_itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestind]
        itr_times['verbose_time'] = time.time() - tstart_verbose
        self.logger.log('verbose time', time.time() - tstart_verbose)

        return scores

    def register_gtruth(self,start_image, last_frames, goal_image):
        """
        :param start_image:
        :param last_frames:
        :param goal_image:
        :return:  returns tradeoff with shape: ncam, ndesig
        """
        last_frames = last_frames[0, self.ncontxt -1]

        warped_image_start_l, warped_image_goal_l, start_warp_pts_l, goal_warp_pts_l, warperrs_l = [], [], [], [], []

        if 'pred_model' in self.gdnconf:
            if self.gdnconf['pred_model'] == MulltiviewTestGDN:
                warped_image_start, _, start_warp_pts = self.goal_image_warper(last_frames[None], start_image[None])
                if 'goal' in self.policyparams['register_gtruth']:
                    warped_image_goal, _, goal_warp_pts = self.goal_image_warper(last_frames[None], goal_image[None])
            else:
                raise NotImplementedError
        else:
            for n in range(self.ncam):  # if a shared goal_image warper is used for all views
                warped_image_goal, goal_warp_pts = None, None
                warped_image_start, _, start_warp_pts = self.goal_image_warper(last_frames[n][None], start_image[n][None])
                if 'goal' in self.policyparams['register_gtruth']:
                    warped_image_goal, _, goal_warp_pts = self.goal_image_warper(last_frames[n][None], goal_image[n][None])
                warped_image_start_l.append(warped_image_start)
                warped_image_goal_l.append(warped_image_goal)
                start_warp_pts_l.append(start_warp_pts)
                goal_warp_pts_l.append(goal_warp_pts)
            warped_image_start = np.stack(warped_image_start_l, 0)
            warped_image_goal = np.stack(warped_image_goal_l, 0)
            start_warp_pts = np.stack(start_warp_pts_l, 0)
            goal_warp_pts = np.stack(goal_warp_pts_l, 0)

        desig_pix_l = []

        imheight, imwidth = goal_image.shape[1:3]
        for n in range(self.ncam):
            start_warp_pts = start_warp_pts.reshape(self.ncam, imheight, imwidth, 2)
            warped_image_start = warped_image_start.reshape(self.ncam, imheight, imwidth, 3)
            if 'goal' in self.policyparams['register_gtruth']:
                goal_warp_pts = goal_warp_pts.reshape(self.ncam, imheight, imwidth, 2)
                warped_image_goal = warped_image_goal.reshape(self.ncam, imheight, imwidth, 3)
            else:
                goal_warp_pts = None
                warped_image_goal = None
            warperr, desig_pix = self.get_warp_err(n, start_image, goal_image, start_warp_pts, goal_warp_pts, warped_image_start, warped_image_goal)
            warperrs_l.append(warperr)
            desig_pix_l.append(desig_pix)

        self.desig_pix = np.stack(desig_pix_l, axis=0).reshape(self.ncam, self.ndesig, 2)

        warperrs = np.stack(warperrs_l, 0)    # shape: ncam, ntask, r

        tradeoff = (1 / warperrs)
        normalizers = np.sum(np.sum(tradeoff, 0, keepdims=True), 2, keepdims=True)
        tradeoff = tradeoff / normalizers
        tradeoff = tradeoff.reshape(self.ncam, self.ndesig)

        self.plan_stat['tradeoff'] = tradeoff
        self.plan_stat['warperrs'] = warperrs.reshape(self.ncam, self.ndesig)
        return warped_image_start, warped_image_goal, tradeoff

    def get_warp_err(self, icam, start_image, goal_image, start_warp_pts, goal_warp_pts, warped_image_start, warped_image_goal):
        r = len(self.policyparams['register_gtruth'])
        warperrs = np.zeros((self.ntask, r))
        desig = np.zeros((self.ntask, r, 2))
        for p in range(self.ntask):
            if 'image_medium' in self.agentparams:
                pix_t0 = self.desig_pix_t0_med[icam, p]
                goal_pix = self.goal_pix_med[icam, p]
                self.logger.log('using desig goal pix medium')
            else:
                pix_t0 = self.desig_pix_t0[icam, p]     # desig_pix_t0 shape: icam, ndesig, 2
                goal_pix = self.goal_pix_sel[icam, p]
                # goal_image = cv2.resize(goal_image, (self.agentparams['image_width'], self.agentparams['image_height']))

            if 'start' in self.policyparams['register_gtruth']:
                desig[p, 0] = np.flip(start_warp_pts[icam][pix_t0[0], pix_t0[1]], 0)
                warperrs[p, 0] = np.linalg.norm(start_image[icam][pix_t0[0], pix_t0[1]] -
                                                warped_image_start[icam][pix_t0[0], pix_t0[1]])

            if 'goal' in self.policyparams['register_gtruth']:
                desig[p, 1] = np.flip(goal_warp_pts[icam][goal_pix[0], goal_pix[1]], 0)
                warperrs[p, 1] = np.linalg.norm(goal_image[icam][goal_pix[0], goal_pix[1]] -
                                                warped_image_goal[icam][goal_pix[0], goal_pix[1]])

        if 'image_medium' in self.agentparams:
            desig = desig * self.agentparams['image_height']/ self.agentparams['image_medium'][0]
        return warperrs, desig

    def publish_sawyer(self, gen_distrib, gen_images, scores):
        sorted_inds = scores.argsort()
        bestind = sorted_inds[0]
        middle = sorted_inds[sorted_inds.shape[0] / 2]
        worst = sorted_inds[-1]
        sel_ind = [bestind, middle, worst]
        # t, r, c, 3
        gen_im_l = []
        gen_distrib_l = []
        gen_score_l = []
        for ind in sel_ind:
            gen_im_l.append(np.stack([im[ind] for im in gen_images], axis=0).flatten())
            gen_distrib_l.append(np.stack([d[ind] for d in gen_distrib], axis=0).flatten())
            gen_score_l.append(scores[ind])
        gen_im_l = np.stack(gen_im_l, axis=0).flatten()
        gen_distrib_l = np.stack(gen_distrib_l, axis=0).flatten()
        gen_score_l = np.array(gen_score_l, dtype=np.float32)
        self.gen_image_publisher.publish(gen_im_l)
        self.gen_pix_distrib_publisher.publish(gen_distrib_l)
        self.gen_score_publisher.publish(gen_score_l)
        return bestind

    def MSE_based_score(self, gen_images, goal_image):

        gen_images = np.stack(gen_images, 1)
        sq_diff = np.square(gen_images - goal_image[:,None])
        if 'goal_mask' in self.agentparams:
            sq_diff *= self.goal_mask[None,None, :,:, None]
        mean_sq_diff = np.mean(sq_diff.reshape([sq_diff.shape[0],sq_diff.shape[1],-1]), -1)

        per_time_multiplier = np.ones([1, gen_images.shape[1]])
        per_time_multiplier[:, -1] = self.policyparams['finalweight']
        return np.sum(mean_sq_diff * per_time_multiplier, axis=1)


    def flow_based_score(self, gen_images, goal_image):
        flow_fields = np.zeros([self.M, self.seqlen-1, self.img_height, self.img_width, 2])
        warped_images = np.zeros([self.M, self.seqlen-1, self.img_height, self.img_width, 3])
        warp_pts_l = []

        for tstep in range(self.seqlen - 1):
            gdn_start = time.time()
            if 'warp_goal_to_pred' in self.policyparams:
                warped_image, flow_field, warp_pts = self.goal_image_warper(goal_image, gen_images[tstep])
            else:
                warped_image, flow_field, warp_pts = self.goal_image_warper(gen_images[tstep], goal_image)
            self.logger.log('time for gdn forward pass {}'.format(time.time() - gdn_start))

            flow_fields[:, tstep] = flow_field
            warped_images[:, tstep] = warped_image
            warp_pts_l.append(warp_pts)

        t_fs1 = time.time()
        scores = compute_warp_cost(self.logger, self.policyparams, flow_fields, self.goal_pix, warped_images, goal_image, self.goal_mask)
        self.logger.log('t_fs1 {}'.format(time.time() - t_fs1))

        return flow_fields, scores, warp_pts_l, warped_images

    def compute_trade_off_cost(self, gen_distrib):
        """
        :param gen_distrib:  shape: b, t, ncam, r, c, p
        :return:
        """
        t_mult = np.ones([self.seqlen - self.netconf['context_frames']])
        t_mult[-1] = self.policyparams['finalweight']

        gen_distrib = gen_distrib.copy()
        #normalize prob distributions
        psum = np.sum(np.sum(gen_distrib, axis=3), 3)
        gen_distrib /= psum[:,:,:, None, None, :]

        if self.policyparams['trade_off_views'] == 'clip_psum':
            psum = np.clip(psum, 0, 1.)

        scores_perdesig_l = []
        tradeoff_l = []
        for p in range(self.ndesig):
            scores_perdesig = np.zeros([self.M, self.seqlen - self.ncontxt, self.ncam])
            for icam in range(self.ncam):
                distance_grid = self.get_distancegrid(self.goal_pix[icam, p])
                scores_perdesig[:, :, icam] = np.sum(np.sum(gen_distrib[:,:,icam,:,:,p]*distance_grid[None, None], axis=2),2)

            tradeoff = psum[...,p]/np.sum(psum[...,p], axis=2)[...,None]        # compute tradeoffs
            tradeoff_l.append(tradeoff)

            scores_perdesig = scores_perdesig*tradeoff
            scores_perdesig = np.sum(scores_perdesig, 2)      #sum over images

            scores_perdesig *= t_mult[None]
            scores_perdesig = np.sum(scores_perdesig, 1)      # sum over timesteps

            scores_perdesig_l.append(scores_perdesig)
        scores = np.stack(scores_perdesig_l, 1)
        tradeoff = np.stack(tradeoff_l, -1)
        return scores, tradeoff


    def calc_scores(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """
        assert len(gen_distrib.shape) == 4
        t_mult = np.ones([self.seqlen - self.netconf['context_frames']])
        t_mult[-1] = self.policyparams['finalweight']

        gen_distrib = gen_distrib.copy()
        #normalize prob distributions
        if normalize:
            gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
        gen_distrib *= distance_grid[None, None]
        scores = np.sum(np.sum(gen_distrib, axis=2),2)
        self.cost_perstep[:,icam, idesig] = scores
        scores *= t_mult[None]
        scores = np.sum(scores, axis=1)/np.sum(t_mult)
        return scores

    def get_distancegrid(self, goal_pix):
        distance_grid = np.empty((self.img_height, self.img_width))
        for i in range(self.img_height):
            for j in range(self.img_width):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        self.logger.log('making distance grid with goal_pix', goal_pix)
        # plt.imshow(distance_grid, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
        # plt.show()
        return distance_grid

    def make_input_distrib(self, itr):
        if 'predictor_propagation' in self.policyparams:  # using the predictor's DNA to propagate, no correction
            input_distrib = self.get_recinput(itr, self.rec_input_distrib, self.desig_pix)
        else:
            input_distrib = self.switch_on_pix(self.desig_pix)
        return input_distrib

    def get_recinput(self, itr, rec_input_distrib, desig):
        ctxt = self.netconf['context_frames']
        if len(rec_input_distrib) < ctxt:
            input_distrib = self.switch_on_pix(desig)
            if itr == 0:
                rec_input_distrib.append(input_distrib[:, 0])
        else:
            input_distrib = [rec_input_distrib[c] for c in range(-ctxt, 0)]
            input_distrib = np.stack(input_distrib, axis=1)
        return input_distrib


    def act(self, traj, t, desig_pix=None, goal_pix=None, goal_image=None, goal_mask=None, curr_mask=None):
        """
        Return a random action for a state.
        Args:
            traj: trajectory object
                if performing highres tracking traj.images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """

        if 'register_gtruth' in self.policyparams:
            self.goal_pix_sel = np.array(goal_pix).reshape((self.ncam, self.ntask, 2))
            num_reg_images = len(self.policyparams['register_gtruth'])
            assert self.ndesig == num_reg_images*self.ntask
            self.goal_pix = np.tile(self.goal_pix_sel[:,:,None,:], [1,1,num_reg_images,1])  # copy along r: shape: ncam, ntask, r
            self.goal_pix = self.goal_pix.reshape(self.ncam, self.ndesig, 2)
        else:
            self.desig_pix = np.array(desig_pix).reshape((self.ncam, self.ndesig, 2))
            self.goal_pix = np.array(goal_pix).reshape((self.ncam, self.ndesig, 2))
        if 'image_medium' in self.agentparams:
            self.goal_pix_med = (self.goal_pix * self.agentparams['image_medium'][0] / self.agentparams['image_height']).astype(np.int)

        super(CEM_Controller_Vidpred).act(traj,t)
