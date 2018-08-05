from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.base_sawyer_mujoco_env import BaseSawyerMujocoEnv
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.demonstration_envs.pick_place_env import PickPlaceEnv
import copy


class VanillaSawyerMujocoEnv(BaseSawyerMujocoEnv):
    def __init__(self, env_params, reset_state=None):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(env_params, reset_state)
        self._adim, self._sdim = self._base_adim, self._base_sdim

    def _init_dynamics(self):
        return

    def _next_qpos(self, action):
        return self._previous_target_qpos * self.mode_rel + action


class PickPlaceDemo(PickPlaceEnv, VanillaSawyerMujocoEnv):
    pass