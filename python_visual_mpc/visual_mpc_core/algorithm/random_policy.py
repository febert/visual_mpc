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

            if 'discrete_gripper' in self.agentparams:
                self.actions = discretize_gripper(self.actions, self.agentparams['discrete_gripper'])

            if 'no_action_bound' not in self.policyparams:
                self.actions = truncate_movement(self.actions, self.policyparams)

        return self.actions[t]

    def finish(self):
        pass

class RandomPickPolicy(Randompolicy):
    def act(self, traj, t, init_model = None, goal_ee_pose = None, agentparams = None, goal_image = None):
        repeat = self.policyparams['repeats']
        assert self.agentparams['T'] == self.naction_steps * repeat and self.naction_steps >= 3

        if t == 0:
            mean = np.zeros((self.naction_steps, self.adim))

            target_object = np.random.randint(self.agentparams['num_objects']) #selects a random object to pick
            traj.desig_pos = traj.Object_pose[0, target_object, :2].copy()
       
            object_xy = traj.Object_pose[0, target_object, :2] / repeat

            mean[0] = np.array([object_xy[0], object_xy[1], self.agentparams.get('ztarget', 0.13) / repeat, 0, -1]) #mean action goes toward object
            mean[1] = np.array([0, 0, (-0.08 - self.agentparams.get('ztarget', 0.13)) / repeat, 0, -1]) #mean action swoops down to pick object
            mean[2] = np.array([0, 0, (-0.08 - self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1]) #mean action gripper grasps object
            mean[3] = np.array([0, 0, (0.08 + self.agentparams.get('ztarget', 0.13)) / repeat, 0, 1])  #mean action lifts hand up

            sigma = construct_initial_sigma(self.policyparams)
            
            self.actions = np.random.multivariate_normal(mean.reshape(-1), sigma).reshape(self.naction_steps, self.adim)
            self.actions = np.repeat(self.actions, repeat, axis = 0)

            if 'discrete_adim' in self.agentparams:
                self.actions = discretize(self.actions, self.agentparams['discrete_adim'])

            if 'discrete_gripper' in self.agentparams:
                self.actions = discretize_gripper(self.actions, self.agentparams['discrete_gripper'])

            if 'no_action_bound' not in self.policyparams:
                self.actions = truncate_movement(self.actions, self.policyparams)

        return self.actions[t]

def discretize_gripper(actions, gripper_ind):
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
