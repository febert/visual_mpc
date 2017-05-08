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

from PIL import Image
import pdb


class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor = None):
        Policy.__init__(self)
        self.agentparams = copy.deepcopy(AGENT_MUJOCO)
        self.agentparams.update(ag_params)
        self.policyparams = policyparams

        self.t = None

        if self.policyparams['low_level_ctrl']:
            self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])

        self.model = mujoco_py.MjModel(self.agentparams['filename'])

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        if 'use_first_plan' in self.policyparams:
            self.use_first_plan = self.policyparams['use_first_plan']
        else: self.use_first_plan = True

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.use_net = self.policyparams['usenet']
        self.action_list = []

        if self.use_net:
            hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
            self.netconf = hyperparams.configuration

        self.naction_steps = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']
        self.M = self.netconf['batch_size']
        assert self.naction_steps * self.repeat == self.netconf['sequence_length']
        self.predictor = predictor

        self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.naction_steps * self.repeat)]
        self.gtruth_states = np.zeros((self.naction_steps * self.repeat, self.M, 4))

        # the full horizon is actions*repeat
        # self.action_cost_mult = 0.00005
        self.action_cost_mult = 0

        self.adim = 4  # action dimensions: deltax, delty, close_nstep, hold_nstep
        self.initial_std = policyparams['initial_std']

        gofast = True
        self.viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                         init_height=480, go_fast=gofast)
        self.viewer.start()
        self.viewer.set_model(self.model)
        self.viewer.cam.camid = 0

        self.small_viewer = mujoco_py.MjViewer(visible=True, init_width=64,
                                         init_height=64, go_fast=gofast)
        self.small_viewer.start()
        self.small_viewer.set_model(self.model)
        self.small_viewer.cam.camid = 0


        self.init_model = []


        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.naction_steps, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.naction_steps, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)

        self.mean =None
        self.sigma =None
        self.goal_image = None


    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.naction_steps * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_mult
        return actions_costs


    def discretize(self, actions):
        for b in range(self.M):
            for a in range(self.naction_steps):
                actions[b, a, 2] = np.ceil(actions[b, a, 2])
                if actions[b, a, 2] < 0:
                    actions[b, a, 2] = 0
                if actions[b, a, 2] > 4:
                    actions[b, a, 2] = 4

                actions[b, a, 3] = np.ceil(actions[b, a, 3])
                if actions[b, a, 3] < 0:
                    actions[b, a, 3] = 0
                if actions[b, a, 3] > 4:
                    actions[b, a, 3] = 4

        return  actions


    def perform_CEM(self,last_frames, last_states, last_action, t):
        # initialize mean and variance

        self.mean = np.zeros(self.adim * self.naction_steps)
        #initialize mean and variance of the discrete actions to their mean and variance used during data collection


        self.sigma = np.diag(np.ones(self.adim * self.naction_steps) * self.initial_std ** 2)

        # dicretize the discrete actions:
        # action_bin_close = range(5)  # values which the action close can take
        # p_close = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        # mean_close, var_close = self.compute_initial_meanvar(p_close)

        # action_bin_up = range(5)
        # p_up = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        # mean_up, var_up = self.compute_initial_meanvar(p_up)

        print '------------------------------------------------'
        print 'starting CEM cylce'


        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr
            t_startiter = datetime.now()

            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.naction_steps, self.adim)
            pdb.set_trace()
            actions = self.discretize(actions)
            pdb.set_trace()

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



    def video_pred(self, last_frames, last_states, actions, itr):


        last_states = np.expand_dims(last_states, axis=0)
        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, 3))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        inf_low_state, gen_images, gen_states = self.predictor( input_images= last_frames,
                                                                input_state=last_states,
                                                                input_actions = actions)
        #evaluate distances to goalstate
        scores = np.zeros(self.netconf['batch_size'])


        for b in range(self.netconf['batch_size']):
            scores[b] = np.linalg.norm(
                (self.goal_image - gen_images[-1][b]).flatten())
            pdb.set_trace()

        # compare prediciton with simulation
        if self.verbose: #and itr == self.policyparams['iterations']-1:
            # print 'creating visuals for best sampled actions at last iteration...'

            file_path = self.netconf['current_dir'] + '/verbose'

            bestindices = scores.argsort()[:self.K]
            bestscores = [scores[ind] for ind in bestindices]

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]

                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist

            self.gtruth_images = [img.astype(np.float) / 255. for img in self.gtruth_images]  #[1:]
            cPickle.dump(best(gen_images), open(file_path + '/gen_image_seq.pkl', 'wb'))
            cPickle.dump(best(self.gtruth_images), open(file_path + '/ground_truth.pkl', 'wb'))
            print 'written files to:' + file_path
            comp_video(file_path, gif_name='check_eval_t{}'.format(self.t))

            f = open(file_path + '/actions_last_iter_t{}'.format(self.t), 'w')
            sorted = scores.argsort()
            for i in range(actions.shape[0]):
                f.write('index: {0}, score: {1}, rank: {2}'.format(i, scores[i],
                                                                   np.where(sorted == i)[0][0]))
                f.write('action {}\n'.format(actions[i]))


        bestindex = scores.argsort()[0]
        if 'store_video_prediction' in self.agentparams and\
                itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestindex]

        if itr == (self.policyparams['iterations']-2):
            self.verbose = True

        return scores


    def act(self, traj, t, init_model= None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
            init_model: mujoco model to initialize from
        """
        self.t = t
        self.init_model = init_model

        if t == 0:
            action = np.zeros(2)
            self.target = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())

        else:

            last_images = traj._sample_images[t-1:t+1]
            last_states = traj.X_Xdot_full[t-1: t+1]
            last_action = self.action_list[-1]

            if self.use_first_plan:
                print 'using actions of first plan, no replanning!!'
                if t == 1:
                    self.perform_CEM(last_images, last_states, last_action, t)
                else:
                    # only showing last iteration
                    self.pred_pos = self.pred_pos[:,-1].reshape((self.M, 1, self.repeat * self.naction_steps, 2))
                    self.rec_target_pos = self.rec_target_pos[:, -1].reshape((self.M, 1, self.repeat * self.naction_steps, 2))
                    self.bestindices_of_iter = self.bestindices_of_iter[-1, :].reshape((1, self.K))
                action = self.bestaction_withrepeat[t - 1]

            else:
                self.perform_CEM(last_images, last_states, last_action, t)
                action = self.bestaction[0]

        self.action_list.append(action)
        print 'timestep: ', t, ' taking action: ', action

        force = action

        return force, self.pred_pos, self.bestindices_of_iter, self.rec_target_pos