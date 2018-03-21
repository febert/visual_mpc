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

        if 'initial_std_grasp' in self.policyparams:
            gr_std = self.policyparams['initial_std_grasp']
        else:
            gr_std = 1.

        if 'initial_std_rot' in self.policyparams:
            rot_std = self.policyparams['initial_std_rot']
        else:
            rot_std = 1.

        if 'initial_std_lift' in self.policyparams:
            lift_std = self.policyparams['initial_std_lift']
        else:
            lift_std = 1.

        diag = []
        for t in range(self.naction_steps):
            if self.adim == 5:
                diag.append(np.array([xy_std**2, xy_std**2, lift_std**2, rot_std**2, gr_std**2]))
            if self.adim == 4:
                diag.append(np.array([xy_std ** 2, xy_std ** 2, lift_std ** 2, gr_std ** 2]))
            elif self.adim == 3:
                diag.append(np.array([xy_std ** 2, xy_std ** 2, lift_std ** 2]))
            elif self.adim == 2:
                diag.append(np.array([xy_std ** 2, xy_std ** 2]))

        diag = np.concatenate(diag, axis=0)
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

            if 'stateful_action' in self.agentparams:
                self.actions[2] = np.ceil(np.abs(self.actions[2])).astype(np.int)


        return self.actions[t]

    def finish(self):
        pass