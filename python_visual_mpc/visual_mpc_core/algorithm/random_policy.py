""" This file defines the linear Gaussian policy class. """
import numpy as np
import scipy

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

from scipy.stats import multivariate_normal
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import construct_initial_sigma
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import truncate_movement

class Randompolicy(Policy):
    """
    Random Policy
    """
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.adim = agentparams['adim']
        self.actions = []

        self.naction_steps = policyparams['nactions']

    def act(self, traj, t, init_model=None, goal_ob_pose=None, agentparams=None, goal_image=None):

        repeat = self.policyparams['repeat']  # repeat the same action to reduce number of repquired timesteps
        assert self.agentparams['T'] == self.naction_steps*repeat

        if t ==0:
            mean = np.zeros(self.adim * self.naction_steps)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = construct_initial_sigma(self.policyparams)

            self.actions = np.random.multivariate_normal(mean, sigma)
            # rv = multivariate_normal(mean, sigma)

            self.actions = self.actions.reshape(self.naction_steps, self.adim)
            self.actions = np.repeat(self.actions, repeat, axis=0)

            if 'discrete_adim' in self.agentparams:
                self.actions = discretize(self.actions, self.agentparams['discrete_adim'])

            if 'no_action_bound' not in self.policyparams:
                self.actions = truncate_movement(self.actions, self.policyparams)

        return self.actions[t]

    def finish(self):
        pass

def discretize(actions, discrete_ind):
    for a in range(actions.shape[0]):
        for ind in discrete_ind:
            actions[a, ind] = np.clip(np.floor(actions[a, ind]), 0, 4)
    return actions
