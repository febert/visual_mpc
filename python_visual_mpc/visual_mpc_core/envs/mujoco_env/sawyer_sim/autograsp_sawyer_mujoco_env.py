from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.base_sawyer_mujoco_env import BaseSawyerMujocoEnv
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.envs.util.action_util import autograsp_dynamics
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.util.sensor_util import is_touching


class AutograspSawyerMujocoEnv(BaseSawyerMujocoEnv):
    def __init__(self, env_params, reset_state=None):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"
        assert env_params.get('finger_sensors', True), "Autograsp requires touch sensors"
        params = copy.deepcopy(env_params)

        if 'autograsp' in params:
            ag_dict = params.pop('autograsp')
            for k in ag_dict:
                params[k] = ag_dict[k]

        super().__init__(params, reset_state)
        self._adim, self._sdim = 4, self._base_sdim

    def _default_hparams(self):
        ag_params = {
            'no_motion_goal': False,
            'reopen': False,
            'zthresh': 0.18,
            'touchthresh': 0.0,
        }

        parent_params = super()._default_hparams()
        parent_params.set_hparam('finger_sensors', True)
        for k in ag_params:
            parent_params.add_hparam(k, ag_params[k])
        return parent_params

    def _init_dynamics(self):
        self._gripper_closed = False
        self._prev_touch = False
        self._goal_reached = False

        if self._hp.no_motion_goal:
            self._last_objects = np.zeros((self._hp.num_objects, 3))
            touch_offset = 0
            if self.finger_sensors:
                touch_offset = 2
            for i in range(self._hp.num_objects):
                self._last_objects[i] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3]
            self._goal_reached = True

    def _next_qpos(self, action):
        assert action.shape[0] == 4, "Action does not match action dimension"

        gripper_z = self._previous_target_qpos[2] + action[2]
        z_thresh = self._hp.zthresh
        reopen = self._hp.reopen

        touch_test = is_touching(self._last_obs['finger_sensors'], self._hp.touchthresh)
        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, gripper_z, z_thresh, reopen,
                                                          touch_test or self._prev_touch)
        self._prev_touch = touch_test
        return target

    def _post_step(self):
        if not self._hp.no_motion_goal:
            finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) > 0
            z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) > 0.15 and self._last_obs['state'][2] > 0.23
            if z_thresholds and finger_sensors_thresh:
                self._goal_reached = True
        else:
            last_objects = np.zeros((self._hp.num_objects, 3))
            touch_offset = 0
            if self.finger_sensors:
                touch_offset = 2

            for i in range(self._hp.num_objects):
                last_objects[i] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3]

            obj_thresh = np.sum(np.abs(last_objects - self._last_objects)) < 1e-2

            if not obj_thresh:
                self._goal_reached = False

    def has_goal(self):
        return True

    def goal_reached(self):
        return self._goal_reached