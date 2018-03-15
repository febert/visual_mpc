""" This file defines the linear Gaussian policy class. """
import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *
import copy
import time
import imp
import cPickle
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from datetime import datetime
import os

from PIL import Image
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import construct_initial_sigma

from pyquaternion import Quaternion

class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        Policy.__init__(self)

        self.agentparams = ag_params
        self.policyparams = policyparams

        self.t = None

        if 'low_level_ctrl' in self.policyparams:
            self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])


        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False


        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.action_list = []

        self.nactions = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']
        self.M = self.policyparams['num_samples']
        self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]

        # the full horizon is actions*repeat
        if 'action_cost_factor' in self.policyparams:
            self.action_cost_mult = self.policyparams['action_cost_factor']
        else: self.action_cost_mult = 0.00005

        self.adim = self.agentparams['adim'] # action dimension
        self.sdim = self.agentparams['sdim'] # action dimension
        self.initial_std = policyparams['initial_std']
        if 'exp_factor' in policyparams:
            self.exp_factor = policyparams['exp_factor']

        self.naction_steps = self.policyparams['nactions']


        self.init_model = None
        #history of designated pixels
        self.desig_pix = []

        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)

        self.rec_input_distrib =[]  # record the input distributions
        self.corr_gen_images = []
        self.corrector = None

        self.mean =None
        self.sigma =None

    def create_sim(self):
        gofast = True
        self.model = mujoco_py.MjModel(self.agentparams['gen_xml_fname'])

        if self.verbose:
            self.viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                             init_height=480, go_fast=gofast)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer.cam.camid = 0

    def reinitialize(self):
        self.use_net = self.policyparams['usenet']
        self.action_list = []
        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]
        self.initial_std = self.policyparams['initial_std']
        # history of designated pixels
        self.desig_pix = []
        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices = []

        self.target = np.zeros(2)

    def finish(self):
        self.viewer.finish()

    def setup_mujoco(self):
        # set initial conditions
        self.model.data.qpos = self.init_model.data.qpos
        self.model.data.qvel = self.init_model.data.qvel
        self.model.step()
        if self.verbose:
            self.viewer.loop_once()

    def eval_action(self):
        abs_distances = []
        abs_angle_dist = []
        qpos_dim = self.sdim / 2  # the states contains pos and vel
        for i_ob in range(self.agentparams['num_objects']):

            goal_pos = self.goal_obj_pose[i_ob, :3]
            curr_pose = self.model.data.qpos[i_ob * 7 + qpos_dim:(i_ob+1) * 7 + qpos_dim].squeeze()
            curr_pos = curr_pose[:3]

            abs_distances.append(np.linalg.norm(goal_pos - curr_pos))

            goal_quat = Quaternion(self.goal_obj_pose[i_ob, 3:])
            curr_quat = Quaternion(curr_pose[3:])
            diff_quat = curr_quat.conjugate*goal_quat
            abs_angle_dist.append(np.abs(diff_quat.radians))

        return np.sum(np.array(abs_distances)), np.sum(np.array(abs_angle_dist))

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.nactions * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_mult
        return actions_costs

    def perform_CEM(self, t):
        # initialize mean and variance

        # initialize mean and variance
        self.mean = np.zeros(self.adim * self.naction_steps)
        # initialize mean and variance of the discrete actions to their mean and variance used during data collection
        self.sigma = construct_initial_sigma(self.policyparams, self.adim)

        print '------------------------------------------------'
        print 'starting CEM cylce'

        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr
            t_startiter = time.time()
            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.naction_steps, self.adim)
            actions = np.repeat(actions, self.repeat, axis=1)

            if 'random_policy' in self.policyparams:
                print 'sampling random actions'
                self.bestaction_withrepeat = actions[0]
                return

            t_start = time.time()

            scores = self.take_mujoco_smp(actions)

            print 'overall time for evaluating actions {}'.format(time.time() - t_start)

            actioncosts = self.calc_action_cost(actions)
            scores += actioncosts

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]

            actions = actions.reshape(self.M, self.naction_steps, self.repeat, self.adim)
            actions = actions[:, :, -1, :]  # taking only one of the repeated actions
            actions_flat = actions.reshape(self.M, self.naction_steps * self.adim)

            self.bestaction = actions[self.indices[0]]
            # print 'bestaction:', self.bestaction

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            self.sigma = np.cov(arr_best_actions, rowvar=False, bias=False)
            self.mean = np.mean(arr_best_actions, axis=0)

            print 'iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]])
            print 'action cost of best action: ', actioncosts[self.indices[0]]

            print 'overall time for iteration {}'.format(time.time() - t_startiter)

    def take_mujoco_smp(self, actions):
        all_scores = np.empty(self.M, dtype=np.float64)
        for smp in range(self.M):
            self.setup_mujoco()
            score = self.sim_rollout(actions[smp])[:, 0]
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self.policyparams['finalweight']
            all_scores[smp] = np.sum(per_time_multiplier*score)
        return all_scores

    def sim_rollout(self, actions):
        costs = []
        self.hf_qpos_l = []
        self.hf_target_qpos_l = []
        for t in range(self.nactions*self.repeat):
            currentaction = actions[t]
            # print 'time ',t, ' target pos rollout: ', roll_target_pos

            if 'posmode' in self.agentparams:  # if the output of act is a positions
                if t == 0:
                    self.prev_target_qpos = self.model.data.qpos[:self.adim].squeeze()
                    self.target_qpos = self.model.data.qpos[:self.adim].squeeze()
                else:
                    self.prev_target_qpos = self.target_qpos
                mask_rel = self.agentparams['mode_rel'].astype(np.float32)  # mask out action dimensions that use absolute control with 0
                self.target_qpos = currentaction.copy() + self.target_qpos * mask_rel
                # print 'current actions', currentaction
                # print 'prev qpos {}'.format(self.prev_target_qpos)
                # print 'target qpos {}'.format(self.target_qpos)

                self.target_qpos = np.clip(self.target_qpos, self.agentparams['targetpos_clip'][0],
                                                             self.agentparams['targetpos_clip'][1])
            else:
                ctrl = currentaction.copy()

            for st in range(self.agentparams['substeps']):
                # self.viewer.loop_once()
                if 'posmode' in self.agentparams:
                    ctrl = self.get_int_targetpos(st, self.prev_target_qpos, self.target_qpos)
                    # if st%10==0:
                        # print "prev qpos {}, targ qpos {}, int {}".format(self.prev_target_qpos, self.target_qpos, ctrl)
                self.model.data.ctrl = ctrl
                self.model.step()
                self.hf_qpos_l.append(self.model.data.qpos)
                self.hf_target_qpos_l.append(ctrl)

                costs.append(self.eval_action())

                if self.verbose:
                    self.viewer.loop_once()

        # self.plot_ctrls()
        return np.stack(costs, axis=0)

    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self.agentparams['substeps']
        return substep/float(self.agentparams['substeps'])*(next - prev) + prev

    def plot_ctrls(self):
        plt.figure()
        # a = plt.gca()
        self.hf_qpos_l = np.stack(self.hf_qpos_l, axis=0)
        self.hf_target_qpos_l = np.stack(self.hf_target_qpos_l, axis=0)
        tmax = self.hf_target_qpos_l.shape[0]
        for i in range(self.adim):
            plt.plot(range(tmax) , self.hf_qpos_l[:,i], label='q_{}'.format(i))
            plt.plot(range(tmax) , self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
            plt.legend()
            plt.show()

    def act(self, traj, t, init_model, goal_obj_pose, agent_params):
        """
        Return a random action for a state.
        Args:
            x_full, xdot_full history of states.
            ref_point: a reference point on the object which shall be moved to a goal
            dref_point: speed of reference point
            t: the current controller's Time step
            init_model: mujoco model to initialize from
        """
        self.agentparams = agent_params
        self.goal_obj_pose = goal_obj_pose
        self.t = t
        self.init_model = init_model

        if t == 0:
            action = np.zeros(self.adim)
            self.create_sim()
        else:
            if 'use_first_plan' in self.policyparams:
                print 'using actions of first plan, no replanning!!'
                if t == 1:
                    self.perform_CEM(t)
                action = self.bestaction_withrepeat[t - 1]
            else:
                self.perform_CEM(t)
                action = self.bestaction[0]

        self.action_list.append(action)
        print 'timestep: ', t, ' taking action: ', action

        if self.verbose:
            self.viewer.finish()
        return action

