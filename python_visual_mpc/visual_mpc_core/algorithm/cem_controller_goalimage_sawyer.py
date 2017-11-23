""" This file defines the linear Gaussian policy class. """
import numpy as np

import copy
import time
import imp
import cPickle
from datetime import datetime

from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *

from PIL import Image
import pdb
from python_visual_mpc.video_prediction.misc.makegifs2 import create_video_pixdistrib_gif
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import matplotlib.pyplot as plt
import collections

from visual_mpc_rospkg.msg import floatarray
from rospy.numpy_msg import numpy_msg
import rospy

class CEM_controller():
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor = None, save_subdir=None):
        print 'init CEM controller'
        self.agentparams = ag_params
        self.policyparams = policyparams

        self.save_subdir = save_subdir
        self.t = None

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.action_list = []
        self.naction_steps = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']

        if self.policyparams['usenet']:
            hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
            self.netconf = hyperparams.configuration
            self.M = self.netconf['batch_size']
            assert self.naction_steps * self.repeat == self.netconf['sequence_length']
        else:
            self.netconf = {}
            self.M = 1

        self.predictor = predictor

        self.ndesig = self.netconf['ndesig']

        self.K = 10  # only consider K best samples for refitting

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
        if self.adim == 4:
            self.discrete_ind = [2, 3]
        elif self.adim == 5:
            self.discrete_ind = [2, 4]

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

        self.dict_ = collections.OrderedDict()

        self.gen_image_publisher = rospy.Publisher('gen_image', numpy_msg(floatarray), queue_size=10)
        self.gen_pix_distrib_publisher = rospy.Publisher('gen_pix_distrib', numpy_msg(floatarray), queue_size=10)

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

    def truncate_movement(self, actions):
        if 'maxshift' in self.policyparams:
            maxshift = self.policyparams['maxshift']
        else:
            maxshift = .09
        actions[:,:,:2] = np.clip(actions[:,:,:2], -maxshift, maxshift)  # clip in units of meters

        if self.adim == 5: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, :, 3] = np.clip(actions[:, :, 3], -maxrot, maxrot)
        return actions

    def construct_initial_sigma(self):
        xy_std = self.policyparams['initial_std']

        if 'initial_std_grasp' in self.policyparams:
            gr_std = self.policyparams['initial_std_grasp']
        else: gr_std = 1.

        if 'initial_std_lift' in self.policyparams:
            lift_std = self.policyparams['initial_std_lift']
        else: lift_std = 1.

        if self.adim == 5:
            if 'initial_std_rot' in self.policyparams:
                rot_std = self.policyparams['initial_std_rot']
            else: rot_std = 1.

        diag = []
        for t in range(self.naction_steps):
            if self.adim == 5:
                diag.append(np.array([xy_std**2, xy_std**2, lift_std**2, rot_std**2, gr_std**2]))
            else:
                diag.append(np.array([xy_std**2, xy_std**2, gr_std**2, lift_std**2]))

        diagonal = np.concatenate(diag, axis=0)
        sigma = np.diag(diagonal)
        return sigma

    def perform_CEM(self,last_frames, last_states, t):
        # initialize mean and variance
        self.mean = np.zeros(self.adim * self.naction_steps)
        #initialize mean and variance of the discrete actions to their mean and variance used during data collection
        self.sigma = self.construct_initial_sigma()

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


            if 'random_policy' in self.policyparams:
                print 'sampling random actions'
                self.bestaction_withrepeat = actions[0]
                return

            t_start = datetime.now()

            if 'multmachine' in self.policyparams:
                scores = self.ray_video_pred(last_frames, last_states, actions, itr)
            else:
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

    def switch_on_pix(self, desig):
        one_hot_images = np.zeros((self.netconf['batch_size'], self.netconf['context_frames'], self.ndesig, 64, 64, 1), dtype=np.float32)
        # switch on pixels
        for p in range(self.ndesig):
            one_hot_images[:, :, p, desig[p, 0], desig[p, 1]] = 1
            print 'using desig pix',desig[p, 0], desig[p, 1]

        return one_hot_images

    def singlepoint_prob_eval(self, gen_pixdistrib):
        print 'using singlepoint_prob_eval'
        scores = np.zeros(self.netconf['batch_size'])
        for t in range(len(gen_pixdistrib)):
            for b in range(self.netconf['batch_size']):
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
                self.rec_input_distrib.append(np.repeat(best_gen_distrib, self.netconf['batch_size'], 0))

        return scores

    def video_pred(self, last_frames, last_states, actions, itr):

        last_states = np.expand_dims(last_states, axis=0)
        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, 3))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        input_distrib = self.make_input_distrib(itr)
        f, axarr = plt.subplots(1, self.ndesig)
        for p in range(self.ndesig):
            axarr[p].imshow(np.squeeze(input_distrib[p][0]), cmap=plt.get_cmap('jet'))
            axarr[p].set_title('input_distrib{}'.format(p), fontsize=8)

        plt.show()
        gen_images, gen_distrib, gen_states, _ = self.predictor(input_images=last_frames,
                                                                input_state=last_states,
                                                                input_actions=actions,
                                                                input_one_hot_images=input_distrib,
                                                                )

        scores_per_task = []
        for p in range(self.ndesig):
            distance_grid = self.get_distancegrid(self.goal_pix[p])
            scores_per_task.append(self.calc_scores(gen_distrib, distance_grid))
            print 'best score of task {}:  {}'.format(p, np.min(scores_per_task[-1]))

        scores = np.sum(np.stack(scores_per_task, axis=1), axis=1)
        #TODO: check dimensions
        pdb.set_trace()
        bestind = scores.argsort()[0]
        for p in range(self.ndesig):
            print 'score {} of best traj: ', scores_per_task[p][bestind]

        # for predictor_propagation only!!
        if 'predictor_propagation' in self.policyparams:
            assert not 'correctorconf' in self.policyparams
            if itr == (self.policyparams['iterations'] - 1):
                # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                bestind = scores.argsort()[0]
                best_gen_distrib = gen_distrib[2][bestind].reshape(1, self.ndesig, 64, 64, 1)
                self.rec_input_distrib.append(np.repeat(best_gen_distrib, self.netconf['batch_size'], 0))

        bestindices = scores.argsort()[:self.K]

        if self.verbose and itr == self.policyparams['iterations']-1:
            # print 'creating visuals for best sampled actions at last iteration...'
            if self.save_subdir != None:
                file_path = self.netconf['current_dir']+ '/'+ self.save_subdir +'/verbose'
            else:
                file_path = self.netconf['current_dir'] + '/verbose'

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]
                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist

            self.dict_['gen_images_t{}'.format(self.t)] = best(gen_images)


            for p in range(self.ndesig):
                self.dict_['gen_distrib{}_t{}'.format(p, self.t)] = best(gen_distrib[:,:,p])

            if not 'no_instant_gif' in self.policyparams:
                t_dict_ = {}
                t_dict_['gen_images_t{}'.format(self.t)] = best(gen_images)
                for p in range(self.ndesig):
                    t_dict_['gen_distrib{}_t{}'.format(p, self.t)] = best(gen_distrib[:,:,p])
                v = Visualizer_tkinter(t_dict_, append_masks=False,
                                       filepath=self.policyparams['current_dir'] + '/verbose',
                                       numex=5)
                v.build_figure()

            best_gen_images = gen_images[bestind]
            best_gen_distrib_seq = gen_distrib[bestind]
            self.gen_image_publisher.publish(best_gen_images)
            self.gen_pix_distrib_publisher.publish(best_gen_distrib_seq)

        if 'store_video_prediction' in self.agentparams and\
                itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestind]

        return scores

    def calc_scores(self, gen_distrib, distance_grid):
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
        return scores

    def get_distancegrid(self, goal_pix):
        distance_grid = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        print 'making distance grid with goal_pix', goal_pix
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
        if self.t < self.netconf['context_frames']:
            input_distrib = self.switch_on_pix(desig)
            if itr == 0:
                rec_input_distrib.append(input_distrib[:, 1])
        else:
            input_distrib = [rec_input_distrib[-2], rec_input_distrib[-1]]
            input_distrib = [np.expand_dims(elem, axis=1) for elem in input_distrib]
            input_distrib = np.concatenate(input_distrib, axis=1)
        return input_distrib

    def act(self, traj, t, desig_pix = None, goal_pix= None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
            init_model: mujoco model to initialize from
        """
        self.t = t
        print 'starting cem at t{}...'.format(t)

        self.desig_pix = np.array(desig_pix).reshape((2, 2))
        if t == 0:
            action = np.zeros(self.agentparams['adim'])
            self.goal_pix = np.array(goal_pix).reshape((2,2))
        else:
            last_images = traj._sample_images[t - 1:t + 1]   # second image shall contain front view
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

                print '########'
                print 'best action sequence: '
                for i in range(self.bestaction.shape[0]):
                    print "t{}: {}".format(i, self.bestaction[i])
                print '########'

        self.action_list.append(action)
        # print 'timestep: ', t, ' taking action: ', action
        return action, self.pred_pos, self.bestindices_of_iter, self.rec_input_distrib
