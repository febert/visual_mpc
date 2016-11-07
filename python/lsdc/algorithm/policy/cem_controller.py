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

class Cem_controller(Policy):
    """
    Random Policy
    """
    def __init__(self, ag_params, policyparams):
        Policy.__init__(self)
        self.agentparams = copy.deepcopy(AGENT_MUJOCO)
        self.agentparams.update(ag_params)
        self.policyparams = policyparams

        self.model = mujoco_py.MjModel(self.agentparams['filename'])
        self._data = {}  #dictionary for storing the data

        self.verbose = False

        if self.verbose:
            self.viewer = mujoco_py.MjViewer(visible=True, init_width=640,
                                             init_height=480, go_fast=False)
            self.viewer.start()
            self.viewer.cam.camid = 0
            self.viewer.set_model(self.model)

    def setup_mujoco(self, init_model):

        # set initial conditions
        self.model.data.qpos = init_model.data.qpos
        self.model.data.qvel = init_model.data.qvel

    def eval_action(self):
        goalpoint = np.array(self.agentparams['goal_point'])
        refpoint = self.model.data.site_xpos[0,:2]
        return np.linalg.norm(goalpoint - refpoint)

    def perform_CEM(self, init_model):

        niter = 5  # number of iterations

        M = 10  # number of samples
        horizon = 5  # MPC horizon
        K = 3  # only consider K best samples for refitting
        adim = 2  # action dimension

        # initialize mean and variance
        mean = np.zeros(2)
        initial_var = 40
        sigma = np.diag(np.ones(2) * initial_var)

        actions = np.zeros((M, adim))
        scores = np.zeros(M)


        for itr in range(niter):

            for smp in range(M):

                self.setup_mujoco(init_model)

                actions[smp] = np.random.multivariate_normal(mean, sigma, 1)
                for tstep in range(horizon):

                    if self.verbose:
                        self.viewer.loop_once()

                        time.sleep(.01)

                    for _ in range(self.agentparams['substeps']):
                        self.model.data.ctrl = actions[smp]
                        self.model.step()  # simulate the model in mujoco

                    # self._data = self.model.data

                scores[smp] = self.eval_action()
                # print scores

            indices = scores.argsort()[:K]
            arr_best_actions = actions[indices]  # only take the K best actions
            sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            mean = np.mean(arr_best_actions, axis= 0)

            bestaction = actions[indices[0]]
            print 'iter {0}, bestscore {1}'.format(itr, scores[indices[0]])

        print 'taking action: ', bestaction

        return bestaction

    def act(self, x, xdot, sample_images, t, init_model= None):
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
        return self.perform_CEM(init_model)