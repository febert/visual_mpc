from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.base_sawyer_mujoco_env import BaseSawyerMujocoEnv
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.envs.util.action_util import autograsp_dynamics
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.util.sensor_util import is_touching

class AutograspSawyerMujocoEnv(BaseSawyerMujocoEnv):
    def __init__(self, env_params):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"
        assert env_params.get('finger_sensors', True), "Autograsp requires touch sensors"
        params = copy.deepcopy(env_params)

        if 'autograsp' in params:
            ag_dict = params.pop('autograsp')
            for k in ag_dict:
                params[k] = ag_dict[k]

        super().__init__(params)
        self._adim, self._sdim = 4, self._base_sdim

    def _default_hparams(self):
        ag_params = {
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

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim, "Action does not match action dimension"
        
        gripper_z = self._previous_target_qpos[2]
        z_thresh = self._params.zthresh
        reopen = self._params.reopen

        touch_test = is_touching(self._last_obs['finger_sensors'], self._params.touchthresh)
        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, gripper_z, z_thresh, reopen,
                                                          touch_test or self._prev_touch)
        self._prev_touch = touch_test
        return target