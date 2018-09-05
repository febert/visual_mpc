from python_visual_mpc.visual_mpc_core.envs.mujoco_env.base_demonstration_env import BaseDemoEnv
import numpy as np
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp


class PickPlaceEnv(BaseDemoEnv):
    """
    Stage 0: Attempting to Grasp
    Stage 5: Attempted Grasp and Failed
    Stage 10: Has successfully grasped and lifted
    Stage 20: Has successfully placed object
    """
    def get_stage(self):
        if self._demo_t == 0:
            self._cur_stage = 0
            self._grasp_attempt = False
            self._obj_init_pos = self._last_obs['object_poses_full']

        print('demo_t {}'.format(self._demo_t))

        if self._cur_stage == 0:
            delta_obj = self._last_obs['object_poses_full'][:, 2] - self._obj_init_pos[:, 2]
            z_thresholds = np.amax(delta_obj) >= 0.05 and self._last_obs['state'][1] > 0.02
            print('check to prog 10, z_threshold {}'.format(z_thresholds))
            if z_thresholds:
                self._cur_stage = 10

        elif self._cur_stage == 10:
            z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) <= 0.
            gripper_stat = self._last_obs['state'][2] > 0.9    # check if gripper is open
            print('check to prog 20 finger_thresh: {}, z_thresh: {}'.format(gripper_stat, z_thresholds))
            if gripper_stat and z_thresholds:
                self._cur_stage = 20
        print('ret_stage: {}'.format(self._cur_stage))
        return self._cur_stage

    def goal_reached(self):
        return self._cur_stage == 20


class CartgripperXZGPickPlace(PickPlaceEnv, CartgripperXZGrasp):
    pass