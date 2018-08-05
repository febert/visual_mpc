from python_visual_mpc.visual_mpc_core.envs.mujoco_env.base_demonstration_env import BaseDemoEnv
import numpy as np


class PickPlaceEnv(BaseDemoEnv):
    def get_stage(self):
        if self._demo_t == 0:
            self._cur_stage = 0
            self._no_touch = False
        print('demo_t {}'.format(self._demo_t))
        if self._cur_stage == 0:
            finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) > 0
            z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) > 0.1 and self._last_obs['state'][2] > 0.2
            print('check to prog 10 finger_thresh: {}, z_threshold {}'.format(finger_sensors_thresh, z_thresholds))
            if finger_sensors_thresh and z_thresholds:
                self._cur_stage = 10

        elif self._cur_stage == 10:
            finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) == 0
            print('check to prog 20 finger_thresh: {}'.format(finger_sensors_thresh))
            if finger_sensors_thresh and not self._no_touch:
                self._no_touch = True
            elif finger_sensors_thresh:
                self._cur_stage = 20
        print('ret_stage: {}'.format(self._cur_stage))
        return self._cur_stage

    def goal_reached(self):
        return self._cur_stage == 20

