import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy


class GelsightManualPolicy(Policy):
    """
    Manual policy: Control testbench actions through the keyboard. Whenever an action is taken, an input command is
    prompted. Valid commands are:
    'w': +x direction
    's': -x direction
    'a': -y direction
    'd': +y direction
    'i': -z direction ('up')
    'k': +z direction ('down')
    't': do nothing and end trajectory (all future actions in this trajectory will become zero actions automatically)
    """

    def __init__(self, agentparams, policyparams, gpu_id, npgu):

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams['adim']
        self.collecting = False

    def _default_hparams(self):
        default_dict = {
            'nactions': 5,
            'repeat': 3,
            'action_bound': True,
            'action_order': [None],
            'initial_std': 0.05,  # std dev. in xy
            'initial_std_lift': 0.15,  # std dev. in xy
            'initial_std_rot': np.pi / 18,
            'initial_std_grasp': 2,
            'type': None,
            'discrete_gripper': False,
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t):
        if t == 0:
            self.collecting = True
        if not self.collecting:
            return {'actions': (0, 0, 0)}
        act_map = {'w': (1, 0, 0), 's': (-1, 0, 0), 'a': (0, -1, 0), 'd': (0, 1, 0), 'i': (0, 0, -1), 'k': (0, 0, 1)}
        ch = input("cmd: ")
        while not ch == 't' and ch not in act_map:
            ch = input("cmd: ")
        if ch == 't': # t triggers end of trajectory
            self.collecting = False
            return {'actions': [0, 0, 0]}
        else:
            if ch in act_map:
                return {'actions': act_map[ch]}
            else:
                raise ValueError('Input not found: ' + ch)

    def process_actions(self):
        if len(self.actions.shape) == 2:
            self.actions = self._process(self.actions)
        elif len(self.actions.shape) == 3:  # when processing batch of actions
            new_actions = []
            for b in range(self.actions.shape[0]):
                new_actions.append(self._process(self.actions[b]))
            self.actions = np.stack(new_actions, axis=0)

    def finish(self):
        pass