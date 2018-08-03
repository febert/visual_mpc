""" This file defines an agent for the MuJoCo simulator environment. """
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from .general_agent import GeneralAgent

class CreateConfigAgent(GeneralAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def rollout(self, policy, i_tr):

        # Take the sample.
        self._init()

        initial_env_obs, reset_state = self.env.reset()
        obs = self._post_process_obs(initial_env_obs, True)

        agent_data, policy_outputs = {}, []
        agent_data['traj_ok'] = True

        for t in range(self._hyperparams['T']):
            self.env.move_arm()
            self.env.move_objects()
            try:
                obs = self._post_process_obs(self.env._get_obs(None))
            except ValueError:
                return {'traj_ok': False}, None, None

        agent_data['reset_state'] = reset_state

        return agent_data, obs, policy_outputs

