from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.base_sawyer_mujoco_env import BaseSawyerMujocoEnv
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.demonstration_envs.pick_place_env import PickPlaceEnv
import copy
import numpy as np

class VanillaSawyerMujocoEnv(BaseSawyerMujocoEnv):
    def __init__(self, env_params, reset_state=None):
        self._hyper = copy.deepcopy(env_params)
        super().__init__(env_params, reset_state)
        self._adim, self._sdim = self._base_adim, self._base_sdim

    def _init_dynamics(self):
        self._goal_reached = False

    def _next_qpos(self, action):
        return self._previous_target_qpos * self.mode_rel + action

    def has_goal(self):
        return self._params.finger_sensors

    def goal_reached(self):
        if not self.has_goal():
            raise NotImplementedError
        return self._goal_reached

    def _post_step(self):
        if not self._params.finger_sensors:
            return

        finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) > 0
        z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) > 0.15 and self._last_obs['state'][2] > 0.23
        if z_thresholds and finger_sensors_thresh:
            self._goal_reached = True


class PickPlaceDemo(PickPlaceEnv, VanillaSawyerMujocoEnv):
    pass