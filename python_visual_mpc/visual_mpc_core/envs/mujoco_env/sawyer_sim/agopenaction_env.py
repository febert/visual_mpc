from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.autograsp_env import AutograspSawyerMujocoEnv
import numpy as np


class AGOpenActionEnv(AutograspSawyerMujocoEnv):
    def __init__(self, env_params):
        if 'open_threshold' in env_params:
            self._threshold = env_params.pop('open_threshold')
        else:
            self._threshold = 0
        super().__init__(env_params)
        self._adim = 5

    def _init_dynamics(self):
        super()._init_dynamics()
        self._goal_reached = False

    def _next_qpos(self, action):
        assert action.shape[0] == 5

        target = super()._next_qpos(action[:-1])
        if action[-1] <= self._threshold:                     #if policy outputs an "open" action then override auto-grasp
            self._gripper_closed = False
            target[-1] = -1

        return target

    def _post_step(self):
        finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) > 0
        z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) > 0.15 and self._last_obs['state'][2] > 0.23
        if z_thresholds and finger_sensors_thresh:
            self._goal_reached = True

    def has_goal(self):
        return True

    def goal_reached(self):
        return self._goal_reached
