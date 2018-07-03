from python_visual_mpc.visual_mpc_core.envs.cartgripper_env.base import BaseCartgripperEnv
import copy


class VanillaCartgripperEnv(BaseCartgripperEnv):
    def __init__(self, env_params):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(**self._hyper)
        self.adim, self.sdim = self._base_adim, self._base_sdim

    def _init_dynamics(self):
        return

    def _next_qpos(self, action):
        assert action.shape[0] == self.adim
        return self._previous_target_qpos * self.mode_rel + action
