""" This file defines the linear Gaussian policy class. """
import pdb
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
    def __init__(self, action_proposal_conf, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.adim = agentparams['adim']
        self.actions = []
        self.naction_steps = policyparams['nactions']
        self.repeat = self.policyparams['repeat']

    def act(self, t):
        assert self.agentparams['T'] == self.naction_steps*self.repeat
        if t == 0:
            mean = np.zeros(5 * self.naction_steps)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = construct_initial_sigma(self.policyparams)
            self.actions = np.random.multivariate_normal(mean, sigma).reshape(self.naction_steps, -1)
            self.process_actions()
        return {'actions': self.actions[t, :self.adim]}

    def process_actions(self):

        if len(self.actions.shape) == 2:
            self.actions = self._process(self.actions)
        elif len(self.actions.shape) == 3:   # when processing batch of actions
            newactions = []
            for b in range(self.actions.shape[0]):
                newactions.append(self._process(self.actions[b]))
            self.actions = np.stack(newactions, axis=0)

    def _process(self, actions):
        if 'discrete_adim' in self.agentparams:
            actions = discretize(actions, self.agentparams['discrete_adim'])
        if 'discrete_gripper' in self.agentparams:
            actions = discretize_gripper(actions, self.agentparams['discrete_gripper'])
        if 'no_action_bound' not in self.policyparams:
            actions = truncate_movement(actions, self.policyparams)
            
        actions = np.repeat(actions, self.repeat, axis=0)
        return actions

    def finish(self):
        pass


class CorrRandompolicy(Randompolicy):
    def __init__(self, action_proposal_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, action_proposal_conf, agentparams, policyparams)

    def act(self, t):
        if t == 0:
            self.sample_actions(1)
        return {'actions': self.actions[0, t]}

    def sample_actions(self, nsamples):
        assert self.repeat == 1
        xy_std = self.policyparams['initial_std']
        diag = [xy_std ** 2, xy_std ** 2]

        if 'initial_std_lift' in self.policyparams:
            diag.append(self.policyparams['initial_std_lift'] ** 2)
        if 'initial_std_rot' in self.policyparams:
            diag.append(self.policyparams['initial_std_rot'] ** 2)
        if 'initial_std_grasp' in self.policyparams:
            diag.append(self.policyparams['initial_std_grasp'] ** 2)

        actions = []
        for d in range(len(diag)):
            var = diag[d]
            mean = np.zeros(self.naction_steps)
            cov = np.diag(np.ones(self.naction_steps)) + \
                  np.diag(np.ones(self.naction_steps - 1), k=1) + \
                  np.diag(np.ones(self.naction_steps - 1), k=-1) + \
                  np.diag(np.ones(self.naction_steps - 2), k=2) + \
                  np.diag(np.ones(self.naction_steps - 2), k=-2)
            sigma = cov * var
            actions.append(np.random.multivariate_normal(mean, sigma, nsamples))

        self.actions = np.stack(actions, axis=-1)
        self.process_actions()


class RandomPickPolicy(Randompolicy):
    def __init__(self, action_proposal_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, action_proposal_conf, agentparams, policyparams)

    def act(self, t, object_poses, state):
        assert self.agentparams['T'] == self.naction_steps * self.repeat and self.naction_steps >= 3
        if t == 0:
            self._desig_pos = self.sample_actions(object_poses, state, 1)
        return {'actions': self.actions[t, :self.adim], 'desig_pos': self._desig_pos}

    def sample_actions(self, object_poses, state, nsamples):
        assert self.adim == 4 or self.adim == 5
        repeat = self.repeat
        mean = np.zeros((self.naction_steps, 5))

        target_object = np.random.randint(object_poses.shape[1])  # selects a random object to pick
        desig_pos = object_poses[0, target_object, :2].copy()

        robot_xy = state[0, :2]
        object_xy = (desig_pos - robot_xy) / repeat

        low = -0.08

        mean[0] = np.array([object_xy[0], object_xy[1], self.agentparams.get('ztarget', 0.13) / repeat, 0,
                            -1])  # mean action goes toward object
        mean[1] = np.array([0, 0, (low - self.agentparams.get('ztarget', 0.13)) / repeat, 0,
                            -1])  # mean action swoops down to pick object
        mean[2] = np.array(
            [0, 0, (low - self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1])  # mean action gripper grasps object
        mean[3] = np.array(
            [0, 0, (-low + self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1])  # mean action lifts hand up

        sigma = construct_initial_sigma(self.policyparams)

        self.actions = np.random.multivariate_normal(mean.reshape(-1), sigma, nsamples).reshape(nsamples, self.naction_steps, 5)
        self.actions = self.actions.squeeze()
        self.process_actions()

        return desig_pos

def discretize_gripper(actions, gripper_ind):
    assert len(actions.shape) == 2
    for a in range(actions.shape[0]):
        if actions[a, gripper_ind] >= 0:
            actions[a, gripper_ind] = 1
        else:
            actions[a, gripper_ind] = -1
    return actions


def discretize(actions, discrete_ind):
    for a in range(actions.shape[0]):
        for ind in discrete_ind:
            actions[a, ind] = np.clip(np.floor(actions[a, ind]), 0, 4)
    return actions
