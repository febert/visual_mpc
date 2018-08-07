from .sub_policies import *
import copy


class StageGraph:
    def __init__(self, agent_params, stage_params):
        pass

    def get_active_policy(self, stage_history, obs=None):
        """
        Gets the environment stage history and returns the current active policy
        as well as an integer denoting the current policy's stage

        :param stage_history:
        :return: ActivePolicy, PolicyIndex
        """
        raise NotImplementedError


class PickAndPlace(StageGraph):
    """
    Stage for a Vanilla Pick and Place Policy
        - Composed of a MoveToObject, Pick, MoveToXYZ, Drop movement
        - If the demonstration finishes before time is up commence a random Wiggle movement
    """
    def __init__(self, agent_params, stage_params):
        super().__init__(agent_params, stage_params)
        policy_params, stage_list = [{} for _ in range(5)], [0, 10, 20, 5]
        if stage_params is not None:
            policy_params, stage_list = stage_params

        self._policies = [WiggleToObject(agent_params, policy_params[0], 0, 0),
                          WiggleAndLift(agent_params, policy_params[1], 0, 0),
                          WiggleToXYZ(agent_params, policy_params[2], 0, 0),
                          WiggleAndPlace(agent_params, policy_params[3], 0, 0),
                          Wiggle(agent_params, policy_params[4], 0, 0)]

        self._env_stages = {k:i for i, k in enumerate(stage_list)}
        self._last_stage, self._last_env_stage = 0, None

    def get_active_policy(self, stage_history, obs=None):
        current_stage = stage_history[-1]
        assert current_stage in self._env_stages.keys(), "Current stage {} not mapped to action policy".format(current_stage)

        if 0 == self._env_stages[current_stage]:
            if self._policies[0].is_done(obs['state'], obs['object_poses_full']) or self._last_stage > 0:
                self._last_stage = 1
            else:
                self._last_stage = 0

        elif 1 == self._env_stages[current_stage]:
            if self._policies[2].is_done(obs['state']) or self._last_stage > 2:
                self._last_stage = 3
            else:
                self._last_stage = 2
        else:
            self._last_stage = 4

        return self._policies[self._last_stage], self._last_stage
