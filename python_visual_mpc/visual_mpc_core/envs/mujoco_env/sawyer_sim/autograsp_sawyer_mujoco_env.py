from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.base_sawyer_mujoco_env import BaseSawyerMujocoEnv
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.envs.util.action_util import autograsp_dynamics
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.util.sensor_util import is_touching

class AutograspSawyerMujocoEnv(BaseSawyerMujocoEnv):
    def __init__(self, env_params):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"

        self._hyper = copy.deepcopy(env_params)
        self._ag_dict = self._hyper.pop('autograsp')
        self._hyper['finger_sensors'] = True
        self._hyper['mode_rel'] = np.array([True, True, True, True, False])

        super().__init__(**self._hyper)
        self._adim, self._sdim = 4, self._base_sdim

    def _init_dynamics(self):
        self._gripper_closed = False
        self._prev_touch = False

    def _next_qpos(self, action):
        assert action.shape[0] == 4
        gripper_z = self._previous_target_qpos[2]
        z_thresh = self._ag_dict['zthresh']
        reopen = 'reopen' in self._ag_dict

        touch_test = is_touching(self._last_obs['finger_sensors'])
        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, gripper_z, z_thresh, reopen,
                                                          touch_test or self._prev_touch)
        self._prev_touch = touch_test
        return target