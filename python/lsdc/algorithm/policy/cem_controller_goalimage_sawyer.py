""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
import mujoco_py
from mujoco_py.mjtypes import *
from lsdc.agent.config import AGENT_MUJOCO
import copy
import time
import imp
import cPickle
from video_prediction.utils_vpred.create_gif import comp_video
from datetime import datetime

from video_prediction.sawyer.create_gif import *

from PIL import Image
import pdb


class CEM_controller():
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor = None):
        print 'init CEM controller'
        self.agentparams = ag_params
        self.policyparams = policyparams

        self.t = None

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.action_list = []

        hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
        self.netconf = hyperparams.configuration

        self.naction_steps = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']
        self.M = self.netconf['batch_size']
        assert self.naction_steps * self.repeat == self.netconf['sequence_length']
        self.predictor = predictor

        self.K = 10  # only consider K best samples for refitting

        # the full horizon is actions*repeat
        # self.action_cost_mult = 0.00005
        if 'action_cost_factor' in self.policyparams:
            self.action_cost_factor = self.policyparams['action_cost_factor']
        else: self.action_cost_factor = 0

        self.adim = 4  # action dimensions: deltax, delty, close_nstep, hold_nstep
        self.initial_std = policyparams['initial_std']


        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.naction_steps, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.naction_steps, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.rec_input_distrib = []  # record the input distributions

        self.target = np.zeros(2)

        self.mean =None
        self.sigma =None
        self.goal_image = None

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
                actions[b, a, 2] = np.floor(actions[b, a, 2])
                if actions[b, a, 2] < 0:
                    actions[b, a, 2] = 0
                if actions[b, a, 2] > 4:
                    actions[b, a, 2] = 4

                actions[b, a, 3] = np.floor(actions[b, a, 3])
                if actions[b, a, 3] < 0:
                    actions[b, a, 3] = 0
                if actions[b, a, 3] > 4:
                    actions[b, a, 3] = 4

        return actions


    def truncate_movement(self, actions):
        actions[:,:,:2] = np.clip(actions[:,:,:2], -.07, .07)  # clip in units of meters
        return actions

    def perform_CEM(self,last_frames, last_states, t):
        # initialize mean and variance

        if 'reuse_mean_cov' not in self.policyparams or t < 2:
            self.mean = np.zeros(self.adim * self.naction_steps)
            #initialize mean and variance of the discrete actions to their mean and variance used during data collection
            self.sigma = np.diag(np.ones(self.adim * self.naction_steps) * self.initial_std ** 2)
            # reducing the variance for goup and close actiondims
            diagonal = copy.deepcopy(np.diag(self.sigma))
            diagonal[2::4] = 1
            diagonal[3::4] = 1
            self.sigma[np.diag_indices_from(self.sigma)] = diagonal

        else:
            print 'reusing mean form last MPC step...'
            mean_old = copy.deepcopy(self.mean)

            self.mean = np.zeros_like(mean_old)
            self.mean[:-self.adim] = mean_old[self.adim:]
            self.mean = self.mean.reshape(self.adim * self.naction_steps)

            sigma_old = copy.deepcopy(self.sigma)
            self.sigma = np.zeros_like(self.sigma)
            self.sigma[0:-self.adim,0:-self.adim] = sigma_old[self.adim:,self.adim: ]
            self.sigma[0:-self.adim, 0:-self.adim] += np.diag(np.ones(self.adim * (self.naction_steps-1)))*(self.initial_std/5)**2
            self.sigma[-1,-1] = 1
            self.sigma[-2,-2] = 1
            self.sigma[-3, -2] = self.initial_std ** 2
            self.sigma[-4, -2] = self.initial_std ** 2
            # self.sigma = np.diag(np.ones(self.adim * self.nactions) * self.initial_std ** 2)

        print '------------------------------------------------'
        print 'starting CEM cylce'

        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr
            t_startiter = datetime.now()

            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.naction_steps, self.adim)
            actions = self.discretize(actions)
            actions = self.truncate_movement(actions)

            actions = np.repeat(actions, self.repeat, axis=1)

            t_start = datetime.now()
            scores = self.video_pred(last_frames, last_states, actions, itr)
            print 'overall time for evaluating actions {}'.format(
                (datetime.now() - t_start).seconds + (datetime.now() - t_start).microseconds / 1e6)

            actioncosts = self.calc_action_cost(actions)
            scores += actioncosts

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]

            actions = actions.reshape(self.M, self.naction_steps, self.repeat, self.adim)
            actions = actions[:,:,-1,:] #taking only one of the repeated actions
            actions_flat = actions.reshape(self.M, self.naction_steps * self.adim)

            self.bestaction = actions[self.indices[0]]
            # print 'bestaction:', self.bestaction

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            self.sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            self.mean = np.mean(arr_best_actions, axis= 0)

            print 'iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]])
            print 'action cost of best action: ', actioncosts[self.indices[0]]

            print 'overall time for iteration {}'.format(
                (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds / 1e6)

    def switch_on_pix(self):
        one_hot_images = np.zeros((self.netconf['batch_size'], self.netconf['context_frames'], 64, 64, 1), dtype=np.float32)
        # switch on pixels
        one_hot_images[:, :, self.desig_pix[0], self.desig_pix[1]] = 1
        return one_hot_images

    def video_pred(self, last_frames, last_states, actions, itr):

        last_states = np.expand_dims(last_states, axis=0)
        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        if 'single_view' in self.netconf:
            img_channels = 3
        else: img_channels = 6
        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, img_channels))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        if 'single_view' in self.netconf:

            if 'use_goalimage' in self.policyparams:
                input_distrib = None
            else:
                input_distrib = self.make_input_distrib(itr)

            gen_images, gen_distrib, gen_states  = self.predictor(input_images=last_frames,
                                                                  input_state=last_states,
                                                                  input_actions=actions,
                                                                  input_one_hot_images=input_distrib)

        #evaluate distances to goalstate
        scores = np.zeros(self.netconf['batch_size'])

        if 'use_goalimage' in self.policyparams:
            for b in range(self.netconf['batch_size']):
                scores[b] = np.linalg.norm((self.goal_image - gen_images[-1][b]).flatten())
        else: # evaluate pixel movement:

            distance_grid = np.empty((64, 64))
            for i in range(64):
                for j in range(64):
                    pos = np.array([i, j])
                    distance_grid[i, j] = np.linalg.norm(self.goal_pix - pos)
            expected_distance = np.zeros(self.netconf['batch_size'])

            if 'rew_all_steps' in self.policyparams:
                for tstep in range(self.netconf['sequence_length'] - 1):
                    t_mult = 1
                    if 'finalweight' in self.policyparams:
                        if tstep == self.netconf['sequence_length'] - 2:
                            t_mult = self.policyparams['finalweight']

                    for b in range(self.netconf['batch_size']):
                        gen = gen_distrib[tstep][b].squeeze() / np.sum(gen_distrib[tstep][b])
                        expected_distance[b] += np.sum(np.multiply(gen, distance_grid)) * t_mult
                scores = expected_distance
            else:
                for b in range(self.netconf['batch_size']):
                    gen = gen_distrib[-1][b].squeeze() / np.sum(gen_distrib[-1][b])
                    expected_distance[b] = np.sum(np.multiply(gen, distance_grid))
                scores = expected_distance


            if 'avoid_occlusions' in self.policyparams:
                occlusion_cfactor = self.policyparams['avoid_occlusions']
                occulsioncost = np.zeros(self.netconf['batch_size'])
                if 'predictor_propagation' in self.policyparams:
                    psum_initval = np.sum(self.rec_input_distrib[-1][0])*self.policyparams['violation_threshold']
                else: psum_initval = self.policyparams['violation_threshold']

                print 'occlusion cost threshold:', psum_initval

                for tstep in range(self.netconf['sequence_length'] - 1):
                    for b in range(self.netconf['batch_size']):
                        occulsioncost[b] =  np.maximum(psum_initval - np.sum(gen_distrib[tstep][b]), 0)*occlusion_cfactor
                        scores[b] += occulsioncost[b]

                print 'occlusion cost of best action', occulsioncost[scores.argsort()[0]]
                print 'occlusion cost of worst action', occulsioncost[scores.argsort()[-1]]
                bestindices = scores.argsort()[:self.K]
                occulsioncost_bestactions = []
                for i in range(self.K):
                    ind = bestindices[i]
                    occulsioncost_bestactions.append(occulsioncost[ind])
                    print 'rank{} , totalcost {}, occlusioncost {}'.format(i, scores[ind], occulsioncost[ind])
                cPickle.dump(occulsioncost_bestactions, open(self.netconf['current_dir'] +
                                    '/verbose/occulsioncost_bestactions.pkl', 'wb'))


        # for predictor_propagation only!!
        if 'predictor_propagation' in self.policyparams:
            assert not 'correctorconf' in self.policyparams
            if itr == (self.policyparams['iterations'] - 1):
                # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                bestind = scores.argsort()[0]
                best_gen_distrib = gen_distrib[2][bestind].reshape(1, 64, 64, 1)
                self.rec_input_distrib.append(np.repeat(best_gen_distrib, self.netconf['batch_size'], 0))

        bestindices = scores.argsort()[:self.K]

        if self.verbose and itr == self.policyparams['iterations']-1:
            # print 'creating visuals for best sampled actions at last iteration...'
            file_path = self.netconf['current_dir'] + '/verbose'

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]
                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist

            cPickle.dump(best(gen_images), open(file_path + '/gen_image_t{}.pkl'.format(self.t), 'wb'))
            if 'use_goalimage' not in self.policyparams:
                cPickle.dump(best(gen_distrib), open(file_path + '/gen_distrib_t{}.pkl'.format(self.t), 'wb'))

            print 'written files to:' + file_path
            if not 'no_instant_gif' in self.policyparams:
                if 'use_goalimage' in self.policyparams:
                    create_video_gif(file_path, self.netconf, t=self.t, n_exp=10, suffix='iter{}_t{}'.format(itr, self.t))
                else:
                    create_video_pixdistrib_gif(file_path, self.netconf, t=self.t, n_exp=10,
                                            suppress_number=True, suffix='iter{}_t{}'.format(itr, self.t))

            # f = open(file_path + '/actions_last_iter_t{}'.format(self.t), 'w')
            # for i in range(actions.shape[0]):
            #     f.write('index: {0}, score: {1}, rank: {2}'.format(i, scores[i],
            #                                                        np.where(sorted == i)[0][0]))
            #     f.write('action {}\n'.format(actions[i]))
            # pdb.set_trace()

        bestindex = scores.argsort()[0]
        if 'store_video_prediction' in self.agentparams and\
                itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestindex]

        return scores

    def make_input_distrib(self, itr):
        if 'predictor_propagation' in self.policyparams:  # using the predictor's DNA to propagate, no correction
            print 'using predictor_propagation'
            if self.t < self.netconf['context_frames']:
                input_distrib = self.switch_on_pix()
                if itr == 0:
                    self.rec_input_distrib.append(input_distrib[:, 1])
            else:
                input_distrib = [self.rec_input_distrib[-2], self.rec_input_distrib[-1]]
                input_distrib = [np.expand_dims(elem, axis=1) for elem in input_distrib]
                input_distrib = np.concatenate(input_distrib, axis=1)
        else:
            input_distrib = self.switch_on_pix()
        return input_distrib

    def act(self, traj, t, desig_pix = None, goal_pix= None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
            init_model: mujoco model to initialize from
        """
        self.t = t

        if t == 0:
            action = np.zeros(4)
            self.desig_pix = desig_pix
            self.goal_pix = goal_pix
        else:
            if 'single_view' in self.netconf:
                last_images = traj._sample_images[t - 1:t + 1]   # second image shall contain front view
            else:
                last_images = traj._sample_images[t-1:t+1]
            last_states = traj.X_full[t-1: t+1]

            if 'use_first_plan' in self.policyparams:
                print 'using actions of first plan, no replanning!!'
                if t == 1:
                    self.perform_CEM(last_images, last_states, t)
                else:
                    # only showing last iteration
                    self.pred_pos = self.pred_pos[:,-1].reshape((self.M, 1, self.repeat * self.naction_steps, 2))
                    self.bestindices_of_iter = self.bestindices_of_iter[-1, :].reshape((1, self.K))
                action = self.bestaction_withrepeat[t - 1]

            else:
                self.perform_CEM(last_images, last_states, t)
                action = self.bestaction[0]

        self.action_list.append(action)
        print 'timestep: ', t, ' taking action: ', action

        force = action

        return force, self.pred_pos, self.bestindices_of_iter, self.rec_input_distrib
