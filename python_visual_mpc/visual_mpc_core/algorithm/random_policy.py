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
    def __init__(self, imitationconf, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.adim = agentparams['adim']
        self.actions = []
        self.naction_steps = policyparams['nactions']
        self.repeat = self.policyparams['repeat']

    def act(self, traj, t, init_model=None, goal_ob_pose=None, agentparams=None, goal_image=None):
        assert self.agentparams['T'] == self.naction_steps*self.repeat
        if t ==0:
            mean = np.zeros(self.adim * self.naction_steps)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = construct_initial_sigma(self.policyparams)
            self.actions = np.random.multivariate_normal(mean, sigma).reshape(self.naction_steps, self.adim)
            self.process_actions()
        return self.actions[t]

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
        if 'z_descend_actions' in self.policyparams:
            actions[self.repeat:, 2] = np.minimum(actions[self.repeat:, 2], -actions[self.repeat:, 2])
        actions = np.repeat(actions, self.repeat, axis=0)
        return actions

    def finish(self):
        pass


class CorrRandompolicy(Randompolicy):
    def __init__(self, imitation_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, imitation_conf, agentparams, policyparams)

    def act(self, traj, t, init_model=None, goal_ob_pose=None, agentparams=None, goal_image=None):
        if t == 0:
            self.sample_actions(traj, 1)
        return self.actions[t]

    def sample_actions(self, traj, nsamples):
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
        return self.actions[t]


class RandomPickPolicy(Randompolicy):
    def __init__(self, imitation_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, imitation_conf, agentparams, policyparams)

    def act(self, traj, t, init_model = None, goal_ee_pose = None, agentparams = None, goal_image = None):
        assert self.agentparams['T'] == self.naction_steps * self.repeat and self.naction_steps >= 3
        if t == 0:
            self.sample_actions(traj, 1)
        return self.actions[t]

    def sample_actions(self, traj, nsamples):
        repeat = self.repeat
        mean = np.zeros((self.naction_steps, self.adim))

        target_object = np.random.randint(traj.Object_pose.shape[1])  # selects a random object to pick
        traj.desig_pos = traj.Object_pose[0, target_object, :2].copy()

        if 'rpn_objects' in self.agentparams:
            robot_xy = traj.endeffector_poses[0, :2]
        else:
            robot_xy = traj.X_full[0, :2]
        object_xy = (traj.Object_pose[0, target_object, :2] - robot_xy) / repeat

        low = self.agentparams['targetpos_clip'][0][2]
        mean[0] = np.array([object_xy[0], object_xy[1], self.agentparams.get('ztarget', 0.13) / repeat, 0,
                            -1])  # mean action goes toward object
        mean[1] = np.array([0, 0, (low - self.agentparams.get('ztarget', 0.13)) / repeat, 0,
                            -1])  # mean action swoops down to pick object
        mean[2] = np.array(
            [0, 0, (low - self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1])  # mean action gripper grasps object
        mean[3] = np.array(
            [0, 0, (-low + self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1])  # mean action lifts hand up

        sigma = construct_initial_sigma(self.policyparams)

        self.actions = np.random.multivariate_normal(mean.reshape(-1), sigma, nsamples).reshape(nsamples, self.naction_steps, self.adim)
        self.actions = self.actions.squeeze()
        self.process_actions()

        return self.actions

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
