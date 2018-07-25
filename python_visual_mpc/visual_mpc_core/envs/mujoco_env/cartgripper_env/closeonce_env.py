from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.base_cartgripper import BaseCartgripperEnv
import copy


class CloseOnceCartgripperEnv(BaseCartgripperEnv):
    def __init__(self, env_params):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(**self._hyper)
        self._adim, self._sdim = self._base_adim, self._base_sdim

    def _init_dynamics(self):
        self._gripper_closed = False

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim

        action_copy = copy.deepcopy(action)
        if action[-1] > 0 or self._gripper_closed:
            self._gripper_closed = True
            action_copy[-1] = 1
        return self._previous_target_qpos * self.mode_rel + action_copy