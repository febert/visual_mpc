""" This file defines the linear Gaussian policy class. """
import pdb
import numpy as np
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import collections
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
if "NO_ROS" not in os.environ:
    from visual_mpc_rospkg.msg import floatarray
    from rospy.numpy_msg import numpy_msg
    import rospy

import pickle as pkl
from collections import OrderedDict

import time
from .utils.cem_controller_utils import construct_initial_sigma, reuse_cov, reuse_mean, truncate_movement, make_blockdiagonal

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

class CEM_Controller_Base(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        """
        :param ag_params:
        :param policyparams:
        :param predictor:
        :param save_subdir:
        :param gdnet: goal-distance network
        """
        self._hp = self._default_hparams()
        self.override_defaults(policyparams)

        self.agentparams = ag_params
        if 'logging_dir' in self.agentparams:
            self.logger = Logger(self.agentparams['logging_dir'], 'cem{}log.txt'.format(self.agentparams['gpu_id']))
        else:
            self.logger = Logger(printout=True)
        self.logger.log('init CEM controller')

        self.t = None

        if self._hp.verbose:
            self.verbose = True
            if isinstance(self._hp.verbose, int):
                self.verbose_freq = self._hp.verbose
            else: self.verbose_freq = 1
        else:
            self.verbose = False
            self.verbose_freq = 1

        self.niter = self._hp.iterations

        self.action_list = []
        self.naction_steps = self._hp.nactions
        self.repeat = self._hp.repeat

        if isinstance(self._hp.num_samples, list):
            self.M = self._hp.num_samples[0]
        else:
            self.M = self._hp.num_samples

        if self._hp.selection_frac:
            self.K = int(np.ceil(self.M*self._hp.selection_frac))
        else:
            self.K = 10  # only consider K best samples for refitting

        #action dimensions:
        # deltax, delty, goup_nstep, delta_rot, close_nstep
        self.adim = self.agentparams['adim']
        self.sdim = self.agentparams['sdim'] # action dimension

        self.indices =[]
        self.mean =None
        self.sigma =None

        self.dict_ = collections.OrderedDict()

        if 'sawyer' in self.agentparams:
            self.gen_image_publisher = rospy.Publisher('gen_image', numpy_msg(floatarray), queue_size=10)
            self.gen_pix_distrib_publisher = rospy.Publisher('gen_pix_distrib', numpy_msg(floatarray), queue_size=10)
            self.gen_score_publisher = rospy.Publisher('gen_score', numpy_msg(floatarray), queue_size=10)

        self.plan_stat = {} #planning statistics

        self.warped_image_goal, self.warped_image_start = None, None

        if self._hp.stochastic_planning:
            self.smp_peract = self._hp.stochastic_planning[0]
        else: self.smp_peract = 1

        self.ncam = 1
        self.ndesig = 1
        self.best_cost_perstep = np.zeros([self.ncam, self.ndesig, self.repeat*self.naction_steps])
        
        
    def _default_hparams(self):
        default_dict = {
            'verbose': False,
            'verbose_every_itr':False,
            'niter': 3,
            'num_samples': 200,
            'selection_frac': False, # specifcy which fraction of best samples to use to compute mean and var for next CEM iteration
            'discrete_ind':None,
            'reuse_mean':False,
            'reuse'
            'reuse_cov':False,
            'stochastic_planning':False,
            'rejection_sampling':True,
            'reuse_cov':False,
            'cov_blockdiag':False,
            'smooth_cov':False,
            'iterations': 3,
            'nactions': 5,
            'repeat': 3,
            'no_action_bound': False,
            'initial_std': 0.05,   #std dev. in xy
            'initial_std_lift': 0.15,   #std dev. in xy
            'initial_std_rot': np.pi / 18,
            'finalweight':10,
            'use_first_plan':False,
            'replan_interval':1,
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self):
        self.plan_stat = {} #planning statistics
        self.indices =[]
        self.action_list = []


    def discretize(self, actions):
        """
        discretize and clip between 0 and 4
        :param actions:
        :return:
        """
        for b in range(self.M):
            for a in range(self.naction_steps):
                for ind in self.discrete_ind:
                    actions[b, a, ind] = np.clip(np.floor(actions[b, a, ind]), 0, 4)
        return actions

    def perform_CEM(self):
        self.logger.log('starting cem at t{}...'.format(self.t))
        timings = OrderedDict()
        t = time.time()
        if not self._hp.reuse_cov or self.t < 2:
            self.sigma = construct_initial_sigma(self._hp, self.t)
            self.sigma_prev = self.sigma
        else:
            self.sigma = reuse_cov(self.sigma, self.adim, self._hp)

        if not self._hp.reuse_mean or self.t < 2:
            self.mean = np.zeros(self.adim * self.naction_steps)
        else:
            self.mean = reuse_mean(self.bestaction, self._hp)

        if (self._hp.reuse_mean or self._hp.reuse_cov) and self.t >= 2:
            self.M = self._hp.num_samples[1]
            self.K = int(np.ceil(self.M*self._hp.selection_frac))

        self.bestindices_of_iter = np.zeros((self.niter, self.K))
        self.cost_perstep = np.zeros([self.M, self.ncam, self.ndesig, self.seqlen - self.ncontxt])

        self.logger.log('M {}, K{}'.format(self.M, self.K))
        self.logger.log('------------------------------------------------')
        self.logger.log('starting CEM cylce')
        timings['pre_itr'] = time.time() - t
        for itr in range(self.niter):
            itr_times = OrderedDict()
            self.logger.log('------------')
            self.logger.log('iteration: ', itr)
            t_startiter = time.time()

            if self._hp.rejection_sampling:
                actions = self.sample_actions_rej()
            else:
                actions = self.sample_actions()
            itr_times['action_sampling'] = time.time() - t_startiter
            t_start = time.time()

            scores = self.get_rollouts(actions, itr, itr_times)
            itr_times['vid_pred_total'] = time.time() - t_start
            t = time.time()
            self.logger.log('overall time for evaluating actions {}'.format(time.time() - t_start))

            if self._hp.stochastic_planning:
                actions, scores = self.action_preselection(actions, scores)

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]
            self.plan_stat['scores_itr{}'.format(itr)] = scores
            self.plan_stat['bestscore_itr{}'.format(itr)] = scores[self.indices[0]]
            if hasattr(self, 'best_cost_perstep'):
                self.plan_stat['best_cost_perstep'] = self.best_cost_perstep

            actions_flat = self.post_process_actions(actions)
            self.bestaction = actions[self.indices[0]]

            self.fit_gaussians(actions_flat)

            self.logger.log('iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]]))
            self.logger.log('overall time for iteration {}'.format(time.time() - t_startiter))
            itr_times['post_pred'] = time.time() - t
            timings['itr{}'.format(itr)] = itr_times

        pkl.dump(timings, open('{}/timings_CEM_{}.pkl'.format(self.agentparams['record'], self.t), 'wb'))

    def fit_gaussians(self, actions_flat):
        arr_best_actions = actions_flat[self.indices]  # only take the K best actions
        self.sigma = np.cov(arr_best_actions, rowvar=False, bias=False)
        if self._hp.cov_blockdiag:
            self.sigma = make_blockdiagonal(self.sigma, self.naction_steps, self.adim)
        if self._hp.smooth_cov:
            self.sigma = 0.5 * self.sigma + 0.5 * self.sigma_prev
            self.sigma_prev = self.sigma
        self.mean = np.mean(arr_best_actions, axis=0)

    def post_process_actions(self, actions):
        num_ex = self.M // self.smp_peract
        actions = actions.reshape(num_ex, self.naction_steps, self.repeat, self.adim)
        actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(num_ex, self.naction_steps * self.adim)
        self.bestaction = actions[self.indices[0]]
        return actions_flat

    def sample_actions(self):
        actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
        actions = actions.reshape(self.M, self.naction_steps, self.adim)
        if self.discrete_ind != None:
            actions = self.discretize(actions)

        if self._hp.no_action_bound:
            actions = truncate_movement(actions, self._hp)
        actions = np.repeat(actions, self.repeat, axis=1)

        return actions

    def sample_actions_rej(self):
        """
        Perform rejection sampling
        :return:
        """
        runs = []
        actions = []

        if self._hp.stochastic_planning:
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
                xy_std = self._hp.initial_std
                lift_std = self._hp.initial_std_lift

                std_fac = 1.5
                if np.any(action_seq[:, :2] > xy_std*std_fac) or \
                        np.any(action_seq[:, :2] < -xy_std*std_fac) or \
                        np.any(action_seq[:, 2] > lift_std*std_fac) or \
                        np.any(action_seq[:, 2] < -lift_std*std_fac):
                    ok = False
                else: ok = True

            runs.append(i)
            actions.append(action_seq)
        actions = np.stack(actions, axis=0)

        if self._hp.stochastic_planning:
            actions = np.repeat(actions,self._hp.stochastic_planning[0], 0)

        self.logger.log('rejection smp max trials', max(runs))
        if self._hp.discrete_ind != None:
            actions = self.discretize(actions)
        actions = np.repeat(actions, self.repeat, axis=1)

        self.logger.log('max action val xy', np.max(actions[:,:,:2]))
        self.logger.log('max action val z', np.max(actions[:,:,2]))
        return actions

    def action_preselection(self, actions, scores):
        actions = actions.reshape((self.M//self.smp_peract, self.smp_peract, self.naction_steps, self.repeat, self.adim))
        scores = scores.reshape((self.M//self.smp_peract, self.smp_peract))
        if self._hp.stochastic_planning[1] == 'optimistic':
            inds = np.argmax(scores, axis=1)
            scores = np.max(scores, axis=1)
        elif self._hp.stochastic_planning[1] == 'pessimistic':
            inds = np.argmin(scores, axis=1)
            scores = np.min(scores, axis=1)
        else: raise ValueError

        actions = [actions[b, inds[b]] for b in range(self.M//self.smp_peract)]
        return np.stack(actions, 0), scores

    def get_rollouts(self, actions, cem_itr, itr_times):
        raise NotImplementedError

    def act(self, t=None, i_tr=None):
        """
        Return a random action for a state.
        Args:
                if performing highres tracking images is highres image
            t: the current controller's Time step
            goal_pix: in coordinates of small image
            desig_pix: in coordinates of small image
        """
        self.i_tr = i_tr
        self.t = t

        if t == 0:
            action = np.zeros(self.agentparams['adim'])
        else:
            if self._hp.use_first_plan:
                self.logger.log('using actions of first plan, no replanning!!')
                if t == 1:
                    self.perform_CEM()
                action = self.bestaction_withrepeat[t]
            elif self._hp.replan_interval:
                if (t-1) % self._hp.replan_interval == 0:
                    self.last_replan = t
                    self.perform_CEM()
                self.logger.log('last replan', self.last_replan)
                self.logger.log('taking action of ', t - self.last_replan)
                action = self.bestaction_withrepeat[t - self.last_replan]
            else:
                self.perform_CEM()
                action = self.bestaction[0]

                self.logger.log('########')
                self.logger.log('best action sequence: ')
                for i in range(self.bestaction.shape[0]):
                    self.logger.log("t{}: {}".format(i, self.bestaction[i]))
                self.logger.log('########')

        self.action_list.append(action)

        self.logger.log("applying action  {}".format(action))

        return {'actions':action, 'plan_stat':self.plan_stat}
