""" This file defines an agent for the MuJoCo simulator environment. """
from .general_agent import GeneralAgent


class CreateConfigAgent(GeneralAgent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def rollout(self, policy, i_tr):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_tr: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
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

