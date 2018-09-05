from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy, get_policy_args
import copy


class DemoPolicy(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._policy_params = copy.deepcopy(policyparams)
        self._stage_graph = self._policy_params['stage_graph'](ag_params, self._policy_params.get('stage_params', None))
        self._agent_params = ag_params

    def act(self, obs, t, i_tr):
        assert 'stage' in obs, 'Demonstration policies required staged environment'
        active_policy, policy_stage = self._stage_graph.get_active_policy(obs['stage'], obs)
        policy_out = active_policy.act(**get_policy_args(active_policy, obs, t, i_tr))
        policy_out['policy_index'] = policy_stage

        return policy_out

    def reset(self):
        self._stage_graph = self._policy_params['stage_graph'](self._agent_params,
                                                               self._policy_params.get('stage_params', None))
