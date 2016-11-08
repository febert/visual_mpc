""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
from lsdc.utility.general_utils import check_shape
import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *
from lsdc.agent.config import AGENT_MUJOCO
import copy
import time
from video_prediction.prediction_train import setup_ctrl

class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        Policy.__init__(self)
        self.agentparams = copy.deepcopy(AGENT_MUJOCO)
        self.agentparams.update(ag_params)
        self.policyparams = policyparams

        self.model = mujoco_py.MjModel(self.agentparams['filename'])
        self._data = {}  #dictionary for storing the data

        self.verbose = False

        self.niter = 5  # number of iterations
        self.M = 10  # number of samples
        self.horizon = 10  # MPC horizon in terms of number of independent consecutive actions
        self.repeat = 1  #repeat the same action to reduce number of repquired timesteps
        # the full horizon is horizon*repeat

        self.K = 3  # only consider K best samples for refitting
        self.adim = 2  # action dimension
        self.initial_var = 40

        if self.verbose:
            self.viewer = mujoco_py.MjViewer(visible=True, init_width=640,
                                             init_height=480, go_fast=False)
            self.viewer.start()
            self.viewer.cam.camid = 0
            self.viewer.set_model(self.model)

        self.use_net = False
        if self.use_net:
            self.predictor = setup_ctrl(self.policyparams['netconf'])

    def setup_mujoco(self, init_model):

        # set initial conditions
        self.model.data.qpos = init_model.data.qpos
        self.model.data.qvel = init_model.data.qvel

    def eval_action(self):
        goalpoint = np.array(self.agentparams['goal_point'])
        refpoint = self.model.data.site_xpos[0,:2]
        return np.linalg.norm(goalpoint - refpoint)

    def perform_CEM(self, init_model, last_frames):

        # initialize mean and variance
        mean = np.zeros(self.adim*self.horizon)
        sigma = np.diag(np.ones(self.adim*self.horizon) * self.initial_var)

        actions = np.zeros((self.M, self.horizon*self.adim))
        scores = np.zeros(self.M)

        for itr in range(self.niter):
            for smp in range(self.M):

                self.setup_mujoco(init_model)
                actions[smp] = np.random.multivariate_normal(mean, sigma, 1)
                if not self.use_net:
                    self.sim_rollout(actions, smp)
                else:
                    self.video_pred(last_frames)
                scores[smp] = self.eval_action()

            indices = scores.argsort()[:self.K]
            arr_best_actions = actions[indices]  # only take the K best actions
            sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            mean = np.mean(arr_best_actions, axis= 0)

            bestaction = actions[indices[0], 0: self.adim]
            print 'iter {0}, bestscore {1}'.format(itr, scores[indices[0]])

        print 'taking action: ', bestaction

        return bestaction

    def video_pred(self, last_frames):
        netparams = self.policyparams['netconf']
        sl = netparams['sequence_length']
        con_fr = netparams['context_frames']
        img_w = last_frames.shape[1]

        full_frames = np.concatenate((
            last_frames,
            np.zeros((sl - con_fr, img_w, img_w))
        ), axis= 0)

    def sim_rollout(self, actions, smp):

        for tstep in range(self.horizon):
            currentaction = actions[smp, tstep * self.adim:(tstep + 1) * self.adim]
            for r in range(self.repeat):
                if self.verbose:
                    self.viewer.loop_once()
                    time.sleep(.03)

                for _ in range(self.agentparams['substeps']):
                    self.model.data.ctrl = currentaction
                    self.model.step()  # simulate the model in mujoco

    def act(self, x, xdot, last_frames, t, init_model= None):
        """
        Return a random action for a state.
        Args:
            x: State vector.
            ref_point: a reference point on the object which shall be moved to a goal
            dref_point: speed of reference point
            t: Time step.
            x: position of ball
            xdot: velocity of ball
            init_model: mujoco model to initialize from
        """
        return self.perform_CEM(init_model, last_frames)