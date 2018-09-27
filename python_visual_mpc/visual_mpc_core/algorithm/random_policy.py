""" This file defines the linear Gaussian policy class. """
import pdb
import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_controller_utils import construct_initial_sigma
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_controller_utils import truncate_movement

class Randompolicy(Policy):
    """
    Random Policy
    """
    def __init__(self, agentparams, policyparams, gpu_id, npgu):

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams['adim']

    def _default_hparams(self):
        default_dict = {
            'nactions': 5,
            'repeat': 3,
            'action_bound': True,
            'action_order': [None],
            'initial_std': 0.05,   #std dev. in xy
            'initial_std_lift': 0.15,   #std dev. in xy
            'initial_std_rot': np.pi / 18,
            'initial_std_grasp': 2,
            'type':None,
            'discrete_gripper':False,
        }

        parent_params = super(Randompolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t):
        assert self.agentparams['T'] == self._hp.nactions*self._hp.repeat
        if t == 0:
            mean = np.zeros(self.adim * self._hp.nactions)
            # initialize mean and variance of the discrete actions to their mean and variance used during data collection
            sigma = construct_initial_sigma(self._hp, self.adim)
            self.actions = np.random.multivariate_normal(mean, sigma).reshape(self._hp.nactions, -1)
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
        if self._hp.discrete_gripper:
            actions = discretize_gripper(actions, self._hp)
        if self._hp.action_bound:
            actions = truncate_movement(actions, self._hp)
            
        actions = np.repeat(actions, self._hp.repeat, axis=0)
        return actions

    def finish(self):
        pass


class RandomEpsilonAG(Randompolicy):
    """
    Random Policy
    """
    def __init__(self, agentparams, policyparams, gpu_id, npgu):
        super(RandomEpsilonAG, self).__init__(agentparams, policyparams, gpu_id, npgu)
        assert self.adim == 5, "Action dimension should be 5 (vanilla env)"

    def _default_hparams(self):
        default_dict = {
            'z_thresh': 0.15,
            'epsilon': 0.2
        }
        parent_params = super(RandomEpsilonAG, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t, state, finger_sensors):
        base_action = super(RandomEpsilonAG, self).act(t)['actions'].copy()
        ag_action = -1
        if state[-1, 2] <= self._hp.z_thresh or np.abs(state[-1, -1]) < 0.95:
            ag_action = 1
        if np.random.uniform() < self._hp.epsilon:
            ag_action = -ag_action
            print('eps_rand_grasp_action: {}'.format(ag_action))
        else:
            print('rand_grasp_action: {}'.format(ag_action))
        base_action[-1] = ag_action

        return {'actions': base_action}

class CorrRandompolicy(Randompolicy):
    def __init__(self, action_proposal_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, action_proposal_conf, agentparams, policyparams)

    def act(self, t):
        if t == 0:
            self.sample_actions(1)
        return {'actions': self.actions[0, t]}

    def sample_actions(self, nsamples):
        assert self._hp.repeat == 1
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
            mean = np.zeros(self._hp.nactions)
            cov = np.diag(np.ones(self._hp.nactions)) + \
                  np.diag(np.ones(self._hp.nactions - 1), k=1) + \
                  np.diag(np.ones(self._hp.nactions - 1), k=-1) + \
                  np.diag(np.ones(self._hp.nactions - 2), k=2) + \
                  np.diag(np.ones(self._hp.nactions - 2), k=-2)
            sigma = cov * var
            actions.append(np.random.multivariate_normal(mean, sigma, nsamples))

        self.actions = np.stack(actions, axis=-1)
        self.process_actions()


class RandomPickPolicy(Randompolicy):
    def __init__(self, action_proposal_conf, agentparams, policyparams):  # add imiation_conf to keep compatibility with imitation model
        Randompolicy.__init__(self, action_proposal_conf, agentparams, policyparams)

    def act(self, t, object_poses, state):
        assert self.agentparams['T'] == self._hp.nactions * self.repeat and self._hp.nactions >= 3
        if t == 0:
            self._desig_pos = self.sample_actions(object_poses, state, 1)
        return {'actions': self.actions[t, :self.adim], 'desig_pos': self._desig_pos}

    def sample_actions(self, object_poses, state, nsamples):
        assert self.adim == 4 or self.adim == 5
        repeat = self.repeat
        mean = np.zeros((self._hp.nactions, 5))

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

        self.actions = np.random.multivariate_normal(mean.reshape(-1), sigma, nsamples).reshape(nsamples, self._hp.nactions, 5)
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
