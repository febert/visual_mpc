""" This file defines the linear Gaussian policy class. """
import numpy as np

import os
import copy
import time
import imp
import pickle
from datetime import datetime
import copy
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs

from python_visual_mpc.visual_mpc_core.algorithm.utils.make_visuals import make_cem_visuals
# from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *

import collections
import matplotlib.pyplot as plt
import cv2


if "NO_ROS" not in os.environ:
    from visual_mpc_rospkg.msg import floatarray
    from rospy.numpy_msg import numpy_msg
    import rospy

import time

def standardize_and_tradeoff(flow_sc, warp_sc, flow_warp_tradeoff):
    """
    standardize cost vectors ca and cb, compute new scores weighted by tradeoff factor
    :param ca:
    :param cb:
    :return:
    """
    stand_flow_scores = (flow_sc - np.mean(flow_sc)) / (np.std(flow_sc) + 1e-7)
    stand_ws_costs = (warp_sc - np.mean(warp_sc)) / (np.std(warp_sc) + 1e-7)
    w = flow_warp_tradeoff
    return stand_flow_scores * w + stand_ws_costs * (1 - w)


def compute_warp_cost(policyparams, flow_field, goal_pix=None, warped_images=None, goal_image=None, goal_mask=None):
    """
    :param flow_field:  shape: batch, time, r, c, 2
    :param goal_pix: if not None evaluate flowvec only at position of goal pix
    :return:
    """
    tc1 = time.time()
    flow_mags = np.linalg.norm(flow_field, axis=4)
    print('tc1 {}'.format(time.time() - tc1))

    tc2 = time.time()
    if 'compute_warp_length_spot' in policyparams:
        flow_scores = []
        for t in range(flow_field.shape[1]):
            flow_scores_t = 0
            for ob in range(goal_pix.shape[0]):
                flow_scores_t += flow_mags[:,t,goal_pix[ob, 0], goal_pix[ob, 1]]
            flow_scores.append(np.stack(flow_scores_t))
        flow_scores = np.stack(flow_scores, axis=1)
        print('evaluating at goal point only!!')
    elif goal_mask is not None:
        flow_scores = flow_mags*goal_mask[None, None,:,:]
        #compute average warp-length per per pixel which is part
        # of the object of interest i.e. where the goal mask is 1
        flow_scores = np.sum((flow_scores).reshape([flow_field.shape[0], flow_field.shape[1], -1]), -1)/np.sum(goal_mask)
    else:
        flow_scores = np.mean(np.mean(flow_mags, axis=2), axis=2)

    print('tc2 {}'.format(time.time() - tc2))

    per_time_multiplier = np.ones([1, flow_scores.shape[1]])
    per_time_multiplier[:, -1] = policyparams['finalweight']

    if 'warp_success_cost' in policyparams:
        print('adding warp warp_success_cost')
        if goal_mask is not None:
            diffs = (warped_images - goal_image[:, None])*goal_mask[None, None, :, :, None]

            ws_costs = np.sum(sqdiffs.reshape([flow_field.shape[0], flow_field.shape[1], -1]), axis=-1)/np.sum(goal_mask)
        else:
            ws_costs = np.mean(np.mean(np.mean(np.square(warped_images - goal_image[:,None]), axis=2), axis=2), axis=2)*\
                                        policyparams['warp_success_cost']

        flow_scores = np.sum(flow_scores*per_time_multiplier, axis=1)
        ws_costs = np.sum(ws_costs * per_time_multiplier, axis=1)
        stand_flow_scores = (flow_scores - np.mean(flow_scores)) / (np.std(flow_scores) + 1e-7)
        stand_ws_costs = (ws_costs - np.mean(ws_costs)) / (np.std(ws_costs) + 1e-7)
        w = policyparams['warp_success_cost']
        scores = stand_flow_scores*(1-w) + stand_ws_costs*w
    else:
        scores = np.sum(flow_scores*per_time_multiplier, axis=1)

    print('tcg {}'.format(time.time() - tc1))
    return scores

def construct_initial_sigma(policyparams):
    xy_std = policyparams['initial_std']
    diag = []
    diag += [xy_std**2, xy_std**2]

    if 'initial_std_lift' in policyparams:
        diag.append(policyparams['initial_std_lift'])
    if 'initial_std_rot' in policyparams:
        diag.append(policyparams['initial_std_rot'])
    if 'initial_std_grasp' in policyparams:
        diag.append(policyparams['initial_std_grasp'])

    diag = np.tile(diag, policyparams['nactions'])
    diag = np.array(diag)
    sigma = np.diag(diag)
    return sigma


def truncate_movement(actions, policyparams):
    if 'maxshift' in policyparams:
        maxshift = policyparams['maxshift']
    else:
        maxshift = policyparams['initial_std']*2

    if len(actions.shape) == 3:
        actions[:,:,:2] = np.clip(actions[:,:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] == 5: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, :, 3] = np.clip(actions[:, :, 3], -maxrot, maxrot)

    elif len(actions.shape) == 2:
        actions[:,:2] = np.clip(actions[:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] == 5: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, 3] = np.clip(actions[:, 3], -maxrot, maxrot)
    else:
        raise NotImplementedError

    return actions


def get_mask_trafo_scores(policyparams, gen_distrib, goal_mask):
    scores = []
    bsize = gen_distrib[0].shape[0]
    for t in range(len(gen_distrib)):
        score = np.abs(np.clip(gen_distrib[t], 0,1) - goal_mask[None, None, ..., None])
        score = np.mean(score.reshape(bsize, -1), -1)
        scores.append(score)
    scores = np.stack(scores, axis=1)
    per_time_multiplier = np.ones([1, len(gen_distrib)])
    per_time_multiplier[:, -1] = policyparams['finalweight']
    return np.sum(scores * per_time_multiplier, axis=1)

class CEM_controller():
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
        print('init CEM controller')
        self.agentparams = ag_params
        self.policyparams = policyparams

        self.save_subdir = save_subdir
        self.t = None

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        self.niter = self.policyparams['iterations']

        self.action_list = []
        self.naction_steps = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']


        hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
        self.netconf = hyperparams.configuration
        self.bsize = self.netconf['batch_size']
        self.seqlen = self.netconf['sequence_length']
        self.M = self.bsize
        assert self.naction_steps * self.repeat == self.seqlen

        self.ncontxt = self.netconf['context_frames']
        
        self.predictor = predictor
        self.goal_image_warper = goal_image_warper
        self.goal_image = None

        if 'ndesig' in self.netconf:
            self.ndesig = self.netconf['ndesig']
        else: self.ndesig = None

        self.K = 10  # only consider K best samples for refitting

        self.img_height = self.netconf['img_height']
        self.img_width = self.netconf['img_width']

        # the full horizon is actions*repeat
        # self.action_cost_mult = 0.00005
        if 'action_cost_factor' in self.policyparams:
            self.action_cost_factor = self.policyparams['action_cost_factor']
        else: self.action_cost_factor = 0

        #action dimensions:
        #no rotations:  deltax, delty, close_nstep, goup_nstep;
        # with rotations:  deltax, delty, goup_nstep, delta_rot, close_nstep
        self.adim = self.agentparams['adim']
        self.initial_std = policyparams['initial_std']

        # define which indices of the action vector shall be discretized:
        if 'discrete_adim' in self.agentparams:
            self.discrete_ind = self.agentparams['discrete_adim']
        else:
            self.discrete_ind = None

        # predicted positions
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.naction_steps, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.rec_input_distrib = []  # record the input distributions

        self.target = np.zeros(2)
        self.desig_pix = None

        self.mean =None
        self.sigma =None
        self.goal_image = None

        self.dict_ = collections.OrderedDict()

        if 'sawyer' in self.agentparams:
            self.gen_image_publisher = rospy.Publisher('gen_image', numpy_msg(floatarray), queue_size=10)
            self.gen_pix_distrib_publisher = rospy.Publisher('gen_pix_distrib', numpy_msg(floatarray), queue_size=10)
            self.gen_score_publisher = rospy.Publisher('gen_score', numpy_msg(floatarray), queue_size=10)

        self.goal_mask = None
        self.goal_pix = None

        self.plan_stat = {} #planning statistics

        self.warped_image_goal, self.warped_image_start = None, None
        if 'stochastic_planning' in self.policyparams:
            self.smp_peract = self.policyparams['stochastic_planning'][0]
        else: self.smp_peract = 1

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.naction_steps * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_factor
        return actions_costs

    def discretize(self, actions):
        for b in range(self.M):
            for a in range(self.naction_steps):
                for ind in self.discrete_ind:
                    actions[b, a, ind] = np.clip(np.floor(actions[b, a, ind]), 0, 4)
        return actions


    def perform_CEM(self,last_frames, last_frames_med, last_states, t):
        # initialize mean and variance
        self.mean = np.zeros(self.adim * self.naction_steps)
        #initialize mean and variance of the discrete actions to their mean and variance used during data collection
        self.sigma = construct_initial_sigma(self.policyparams)

        print('------------------------------------------------')
        print('starting CEM cylce')

        for itr in range(self.niter):
            print('------------')
            print('iteration: ', itr)
            t_startiter = time.time()

            if 'rejection_sampling' in self.policyparams:
                actions = self.sample_actions_rej()
            else:
                actions = self.sample_actions()

            if 'random_policy' in self.policyparams:
                print('sampling random actions')
                self.bestaction_withrepeat = actions[0]
                return
            t_start = time.time()

            if 'multmachine' in self.policyparams:
                scores = self.ray_video_pred(last_frames, last_states, actions, itr)
            else:
                scores = self.video_pred(last_frames, last_frames_med, last_states, actions, itr)

            print('overall time for evaluating actions {}'.format(time.time() - t_start))

            actioncosts = self.calc_action_cost(actions)
            scores += actioncosts

            if 'stochastic_planning' in self.policyparams:
                actions, scores = self.action_preselection(actions, scores)

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]
            self.plan_stat['scores_itr{}'.format(itr)] = scores
            self.plan_stat['bestscore_itr{}'.format(itr)] = scores[self.indices[0]]

            actions = actions.reshape(self.M//self.smp_peract, self.naction_steps, self.repeat, self.adim)
            actions = actions[:,:,-1,:] #taking only one of the repeated actions
            actions_flat = actions.reshape(self.M//self.smp_peract, self.naction_steps * self.adim)

            self.bestaction = actions[self.indices[0]]
            # print 'bestaction:', self.bestaction

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            self.sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            self.mean = np.mean(arr_best_actions, axis= 0)

            print('iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]]))
            print('action cost of best action: ', actioncosts[self.indices[0]])

            print('overall time for iteration {}'.format(time.time() - t_startiter))

    def sample_actions(self):
        actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
        actions = actions.reshape(self.M, self.naction_steps, self.adim)
        if self.discrete_ind != None:
            actions = self.discretize(actions)
        if 'no_action_bound' not in self.policyparams:
            actions = truncate_movement(actions, self.policyparams)
        actions = np.repeat(actions, self.repeat, axis=1)
        return actions

    def sample_actions_rej(self):
        """
        Perform rejection sampling
        :return:
        """
        runs = []
        actions = []

        if 'stochastic_planning' in self.policyparams:
            num_distinct_actions = self.M // self.smp_peract
        else:
            num_distinct_actions = self.M

        for i in range(num_distinct_actions):
            ok = False
            i = 0
            while not ok:
                i +=1
                action_seq = np.random.multivariate_normal(self.mean, self.sigma, 1)

                action_seq = action_seq.reshape(self.naction_steps, self.adim)
                xy_std = self.policyparams['initial_std']
                lift_std = self.policyparams['initial_std_lift']

                std_fac = 1.5
                if np.any(action_seq[:, :1] > xy_std*std_fac) or \
                   np.any(action_seq[:, :1] < -xy_std*std_fac) or \
                   np.any(action_seq[:, 2] > lift_std*std_fac) or \
                   np.any(action_seq[:, 2] < -lift_std*std_fac):
                    ok = False
                else: ok = True

            runs.append(i)
            actions.append(action_seq)
        actions = np.stack(actions, axis=0)

        if 'stochastic_planning' in self.policyparams:
            actions = np.repeat(actions,self.policyparams['stochastic_planning'][0], 0)

        print('rejection smp max trials', max(runs))
        if self.discrete_ind != None:
            actions = self.discretize(actions)
        if 'no_action_bound' not in self.policyparams:
            actions = truncate_movement(actions, self.policyparams)
        actions = np.repeat(actions, self.repeat, axis=1)
        return actions

    def action_preselection(self, actions, scores):
        actions = actions.reshape((self.M//self.smp_peract, self.smp_peract, self.naction_steps, self.repeat, self.adim))
        scores = scores.reshape((self.M//self.smp_peract, self.smp_peract))
        if self.policyparams['stochastic_planning'][1] == 'optimistic':
            inds = np.argmax(scores, axis=1)
            scores = np.max(scores, axis=1)
        if self.policyparams['stochastic_planning'][1] == 'pessimistic':
            inds = np.argmin(scores, axis=1)
            scores = np.min(scores, axis=1)

        actions = [actions[b, inds[b]] for b in range(self.M//self.smp_peract)]
        return np.stack(actions, 0), scores

    def switch_on_pix(self, desig):
        one_hot_images = np.zeros((1, self.netconf['context_frames'], self.ndesig, self.img_height, self.img_width, 1), dtype=np.float32)
        desig = np.clip(desig, np.zeros(2).reshape((1, 2)), np.array([self.img_height, self.img_width]).reshape((1, 2)) - 1).astype(np.int)
        # switch on pixels
        for p in range(self.ndesig):
            one_hot_images[:, :, p, desig[p, 0], desig[p, 1]] = 1
            print('using desig pix',desig[p, 0], desig[p, 1])

        return one_hot_images

    def singlepoint_prob_eval(self, gen_pixdistrib):
        print('using singlepoint_prob_eval')
        scores = np.zeros(self.bsize)
        for t in range(len(gen_pixdistrib)):
            for b in range(self.bsize):
                scores[b] -= gen_pixdistrib[t][b,self.goal_pix[0,0], self.goal_pix[0,1]]
        return scores

    def ray_video_pred(self, last_frames, last_states, actions, itr):
        input_distrib = self.make_input_distrib(itr)
        input_distrib = input_distrib[0]
        best_gen_distrib, scores = self.predictor(input_images=last_frames,
                                                  input_states=last_states,
                                                  input_actions=actions,
                                                  input_one_hot_images1=input_distrib,
                                                  goal_pix = self.goal_pix[0])

        if 'predictor_propagation' in self.policyparams:
            # for predictor_propagation only!!
            if itr == (self.policyparams['iterations'] - 1):
                self.rec_input_distrib.append(np.repeat(best_gen_distrib, self.bsize, 0))
        return scores


    def video_pred(self, last_frames, last_frames_med, last_states, actions, cem_itr):
        t_0 = time.time()

        last_states = last_states[None]
        last_frames = last_frames.astype(np.float32, copy=False) / 255.
        last_frames = last_frames[None]

        if 'register_gtruth' in self.policyparams and cem_itr == 0:
            if 'image_medium' in self.agentparams:
                last_frames_med = last_frames_med.astype(np.float32, copy=False) / 255.
                last_frames_med = last_frames_med[None]
                self.start_image = copy.deepcopy(self.traj._image_medium[0]).astype(np.float32) / 255.
                self.warped_image_start, self.warped_image_goal = self.register_gtruth(self.start_image, last_frames_med, self.goal_image)
            else:
                self.start_image = copy.deepcopy(self.traj._sample_images[0]).astype(np.float32) / 255.
                self.warped_image_start, self.warped_image_goal = self.register_gtruth(self.start_image, last_frames, self.goal_image)

        if 'use_goal_image' in self.policyparams and 'comb_flow_warp' not in self.policyparams\
                and 'register_gtruth' not in self.policyparams:
            input_distrib = None
        elif 'masktrafo_obj' in self.policyparams:
            curr_obj_mask = np.repeat(self.curr_obj_mask[None], self.netconf['context_frames'], axis=0).astype(np.float32)
            input_distrib = np.repeat(curr_obj_mask[None], self.bsize, axis=0)[...,None]
        else:
            input_distrib = self.make_input_distrib(cem_itr)

        print('t0 ', time.time() - t_0)
        t_startpred = time.time()
        gen_images, gen_distrib, gen_states, _ = self.predictor(input_images=last_frames,
                                                                input_state=last_states,
                                                                input_actions=actions,
                                                                input_one_hot_images=input_distrib)

        print('time for videoprediction {}'.format(time.time() - t_startpred))

        t_startcalcscores = time.time()
        scores_per_task = []

        flow_fields, warped_images, goal_warp_pts_l = None, None, None
        if 'use_goal_image' in self.policyparams and not 'register_gtruth' in self.policyparams:
            # evaluate images with goal-distance network
            goal_image = np.repeat(self.goal_image[None], self.bsize, axis=0)
            if 'MSE_objective' in self.policyparams:
                scores = self.MSE_based_score(gen_images, goal_image)
            elif 'warp_objective' in self.policyparams:
                flow_fields, scores, goal_warp_pts_l, warped_images = self.flow_based_score(gen_images, goal_image)
            warp_scores = copy.deepcopy(scores)

        if 'masktrafo_obj' in self.policyparams:
            scores = get_mask_trafo_scores(self.policyparams, gen_distrib, self.goal_mask)

        if 'use_goal_image' not in self.policyparams or 'comb_flow_warp' in self.policyparams or 'register_gtruth' in self.policyparams:
            for p in range(self.ndesig):
                distance_grid = self.get_distancegrid(self.goal_pix[p])
                gen_distrib_p = [g[:, p] for g in gen_distrib]
                scores_per_task.append(self.calc_scores(gen_distrib_p, distance_grid))
                print('best flow score of task {}:  {}'.format(p, np.min(scores_per_task[-1])))

            scores = np.sum(np.stack(scores_per_task, axis=1), axis=1)
            bestind = scores.argsort()[0]
            for p in range(self.ndesig):
                print('flow score of best traj for task{}: {}'.format(p, scores_per_task[p][bestind]))

            # for predictor_propagation only!!
            if 'predictor_propagation' in self.policyparams:
                assert not 'correctorconf' in self.policyparams
                if cem_itr == (self.policyparams['iterations'] - 1):
                    # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                    bestind = scores.argsort()[0]
                    best_gen_distrib = gen_distrib[1][bestind].reshape(1, self.ndesig, self.img_height, self.img_width, 1)
                    self.rec_input_distrib.append(best_gen_distrib)

            flow_scores = copy.deepcopy(scores)

        if 'comb_flow_warp' in self.policyparams:
            scores = standardize_and_tradeoff(flow_scores, warp_scores, self.policyparams['comb_flow_warp'])

        print('time to calc scores {}'.format(time.time()-t_startcalcscores))

        bestindices = scores.argsort()[:self.K]

        tstart_verbose = time.time()

        # if self.verbose and cem_itr == self.policyparams['iterations']-1:
        if self.verbose:
            gen_images = make_cem_visuals(self, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images,
                                          gen_states, last_frames, goal_warp_pts_l, scores, self.warped_image_goal,
                                          self.warped_image_start, warped_images)
            if 'sawyer' in self.agentparams:
                bestind = self.publish_sawyer(gen_distrib, gen_images, scores)

        if 'store_video_prediction' in self.agentparams and\
                cem_itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestind]

        print('verbose time', time.time() - tstart_verbose)

        return scores


    def register_gtruth(self, start_image, last_frames, goal_image):
        assert len(self.policyparams['register_gtruth']) == self.ndesig
        # register current image to startimage
        ctxt = self.netconf['context_frames']
        desig_l = []

        current_frame = last_frames[0, ctxt - 1]
        warped_image_start, warped_image_goal = None, None

        if 'image_medium' in self.agentparams:
            pix_t0 = self.desig_pix_t0_med[0]
            goal_pix = self.goal_pix_med[0]
            print('using desig goal pix medium')
        else:
            pix_t0 = self.desig_pix_t0[0]
            goal_pix = self.goal_pix[0]
            goal_image = cv2.resize(goal_image, (self.agentparams['image_width'], self.agentparams['image_height']))

        if 'start' in self.policyparams['register_gtruth']:
            warped_image_start, flow_field, goal_warp_pts = self.goal_image_warper(current_frame[None],
                                                                                  start_image[None])
            desig_l.append(np.flip(goal_warp_pts[0, pix_t0[0], pix_t0[1]], 0))
            # self.plan_stat['start_warp_err'] = np.linalg.norm(start_image[pix_t0[0], pix_t0[1]] -
            #                                                   warped_image_start[0, pix_t0[0], pix_t0[1]])

        if 'goal' in self.policyparams['register_gtruth']:
            warped_image_goal, flow_field, start_warp_pts = self.goal_image_warper(current_frame[None],
                                                                                    goal_image[None])
            desig_l.append(np.flip(start_warp_pts[0, goal_pix[0], goal_pix[1]], 0))
            # self.plan_stat['goal_warp_err'] = np.linalg.norm(self.goal_image[pix_t0[0], pix_t0[1]] -
            #                                                   warped_image_goal[0, pix_t0[0], pix_t0[1]])

        if 'image_medium' in self.agentparams:
            self.desig_pix_med = np.stack(desig_l, 0)
            self.desig_pix = np.stack(desig_l, 0) * self.agentparams['image_height']/ self.agentparams['image_medium'][0]
        else:
            self.desig_pix = np.stack(desig_l, 0)

        return warped_image_start, warped_image_goal

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
        flow_fields = np.zeros([self.bsize, self.seqlen-1, self.img_height, self.img_width, 2])
        warped_images = np.zeros([self.bsize, self.seqlen-1, self.img_height, self.img_width, 3])
        warp_pts_l = []

        for tstep in range(self.seqlen - 1):
            gdn_start = time.time()
            if 'warp_goal_to_pred' in self.policyparams:
                warped_image, flow_field, warp_pts = self.goal_image_warper(goal_image, gen_images[tstep])
            else:
                warped_image, flow_field, warp_pts = self.goal_image_warper(gen_images[tstep], goal_image)
            print('time for gdn forward pass {}'.format(time.time() - gdn_start))

            flow_fields[:, tstep] = flow_field
            warped_images[:, tstep] = warped_image
            warp_pts_l.append(warp_pts)

        t_fs1 = time.time()
        scores = compute_warp_cost(self.policyparams, flow_fields, self.goal_pix, warped_images, goal_image, self.goal_mask)
        print('t_fs1 {}'.format(time.time() - t_fs1))

        return flow_fields, scores, warp_pts_l, warped_images


    def calc_scores(self, gen_distrib, distance_grid):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """

        gen_distrib = np.stack(gen_distrib, 1).squeeze()
        t_mult = np.ones([self.seqlen - self.ncontxt])
        t_mult[-1] = self.policyparams['finalweight']

        #normalize prob distributions
        gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
        gen_distrib *= distance_grid[None, None]
        scores = np.sum(np.sum(gen_distrib, axis=2),2)
        scores *= t_mult[None]
        scores = np.sum(scores, axis=1)
        return scores

    def get_distancegrid(self, goal_pix):
        distance_grid = np.empty((self.img_height, self.img_width))
        for i in range(self.img_height):
            for j in range(self.img_width):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        print('making distance grid with goal_pix', goal_pix)
        # plt.imshow(distance_grid, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
        # plt.show()
        # pdb.set_trace()
        return distance_grid

    def make_input_distrib(self, itr):
        if 'predictor_propagation' in self.policyparams:  # using the predictor's DNA to propagate, no correction
            input_distrib = self.get_recinput(itr, self.rec_input_distrib, self.desig_pix)
        else:
            input_distrib = self.switch_on_pix(self.desig_pix)
        return input_distrib

    def get_recinput(self, itr, rec_input_distrib, desig):
        ctxt = self.netconf['context_frames']
        # if self.t < ctxt:
        if len(rec_input_distrib) < ctxt:
            input_distrib = self.switch_on_pix(desig)
            if itr == 0:
                rec_input_distrib.append(input_distrib[:, 0])
        else:
            # input_distrib = [rec_input_distrib[-2], rec_input_distrib[-1]]
            input_distrib = [rec_input_distrib[c] for c in range(-ctxt, 0)]
            input_distrib = [np.expand_dims(elem, axis=1) for elem in input_distrib]
            input_distrib = np.concatenate(input_distrib, axis=1)
        return input_distrib

    def act(self, traj, t, desig_pix = None, goal_pix= None, goal_image=None, goal_mask = None, curr_mask=None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
        """
        self.goal_mask = goal_mask
        self.desig_pix = np.array(desig_pix).reshape((-1, 2))
        self.goal_pix = np.array(goal_pix).reshape((-1, 2))
        if 'register_gtruth' in self.policyparams:
            self.goal_pix = np.tile(self.goal_pix, [2,1])
        if 'image_medium' in self.agentparams:
            self.goal_pix_med = (self.goal_pix * self.agentparams['image_medium'][0] / self.agentparams['image_height']).astype(np.int)

        self.goal_image = goal_image

        last_images_med = None
        self.curr_obj_mask = curr_mask
        self.traj = traj

        self.t = t
        print('starting cem at t{}...'.format(t))

        if t == 0:
            action = np.zeros(self.agentparams['adim'])
            self.desig_pix_t0 = desig_pix
            if 'image_medium' in self.agentparams:
                self.desig_pix_t0_med = (self.desig_pix * self.agentparams['image_medium'][0]/self.agentparams['image_height']).astype(np.int)
        else:
            ctxt = self.netconf['context_frames']
            last_images = traj._sample_images[t-ctxt+1:t+1]  # same as [t - 1:t + 1] for context 2
            if 'image_medium' in self.agentparams:
                last_images_med = traj._image_medium[t-ctxt+1:t+1]  # same as [t - 1:t + 1] for context 2

            if 'use_vel' in self.netconf:
                last_states = traj.X_Xdot_full[t-ctxt+1:t+1]
            else: last_states = traj.X_full[t-ctxt+1:t+1]

            if 'use_first_plan' in self.policyparams:
                print('using actions of first plan, no replanning!!')
                if t == 1:
                    self.perform_CEM(last_images, last_images_med, last_states, t)
                action = self.bestaction_withrepeat[t - 1]
            else:
                self.perform_CEM(last_images, last_images_med, last_states, t)
                action = self.bestaction[0]

                print('########')
                print('best action sequence: ')
                for i in range(self.bestaction.shape[0]):
                    print("t{}: {}".format(i, self.bestaction[i]))
                print('########')

        self.action_list.append(action)

        return action, self.plan_stat
        # return action, self.bestindices_of_iter, self.rec_input_distrib