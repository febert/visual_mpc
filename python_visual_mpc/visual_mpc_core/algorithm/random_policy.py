""" This file defines the linear Gaussian policy class. """
import numpy as np



class Randompolicy():
    """
    Random Policy
    """
    def __init__(self, agentparams, policyparams):
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.actions = []


    def act(self, traj, t):

        actions = self.policyparams['numactions']  # MPC actions in terms of number of independent consecutive actions
        repeat = self.policyparams['repeats']  # repeat the same action to reduce number of repquired timesteps

        adim = 2  # action dimension
        initial_var = self.policyparams['initial_var'] # before: 40

        assert self.agentparams['T'] == actions*repeat

        if t ==0:
            mean = np.zeros(adim * actions)
            sigma = np.diag(np.ones(adim * actions) * initial_var)
            self.actions = np.random.multivariate_normal(mean, sigma, 1)
            self.actions = self.actions.reshape(actions, 2)
            self.actions = np.repeat(self.actions, repeat, axis=0)

        return self.actions[t], None

    def finish(self):
        pass
