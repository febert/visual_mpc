""" This file defines the linear Gaussian policy class. """
import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

from scipy.stats import multivariate_normal


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

    def construct_initial_sigma(self):
        xy_std = self.policyparams['initial_std']
        diag = []
        diag += [xy_std**2, xy_std**2]

        if 'initial_std_lift' in self.policyparams:
            diag.append(self.policyparams['initial_std_lift'])
        if 'initial_std_rot' in self.policyparams:
            diag.append(self.policyparams['initial_std_rot'])
        if 'initial_std_grasp' in self.policyparams:
            diag.append(self.policyparams['initial_std_grasp'])

        diag = np.tile(diag, self.naction_steps)
        diag = np.array(diag)
        sigma = np.diag(diag)
        return sigma

    def act(self, traj, t, init_model=None, goal_ob_pose=None, agentparams=None):

        repeat = self.policyparams['repeats']  # repeat the same action to reduce number of repquired timesteps
        assert self.agentparams['T'] == self.naction_steps*repeat

        if t ==0:
            mean = np.zeros(self.adim * self.naction_steps)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = self.construct_initial_sigma()

            self.actions = np.random.multivariate_normal(mean, sigma)
            # rv = multivariate_normal(mean, sigma)

            self.actions = self.actions.reshape(self.naction_steps, self.adim)
            self.actions = np.repeat(self.actions, repeat, axis=0)

            if 'discrete_adim' in self.agentparams:
                self.actions = discretize(self.actions, self.agentparams['discrete_adim'])
                # print(self.actions)

        return self.actions[t]

    def finish(self):
        pass

def discretize(actions, discrete_ind):
    for a in range(actions.shape[0]):
        for ind in discrete_ind:
            actions[a, ind] = np.clip(np.floor(actions[a, ind]), 0, 4)
    return actions
