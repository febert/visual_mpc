from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.base_sawyer_env import BaseSawyerEnv
import copy
from python_visual_mpc.visual_mpc_core.envs.util.action_util import autograsp_dynamics


class AutograspSawyerEnv(BaseSawyerEnv):
    def __init__(self, env_params, _=None):
        assert 'mode_rel' not in env_params, "Autograsp sets mode_rel"

        self._hyper = copy.deepcopy(env_params)
        self._ag_dict = self._hyper.pop('autograsp')

        BaseSawyerEnv.__init__(self, self._hyper)
        self._adim, self._sdim = 4, self._base_sdim

    def _init_dynamics(self):
        self._gripper_closed = False
        self._prev_touch = False

    def _next_qpos(self, action):
        assert action.shape[0] == 4      # z dimensions are normalized across robots
        norm_gripper_z = (self._previous_target_qpos[2] - self._low_bound[2]) / \
                         (self._high_bound[2] - self._low_bound[2])
        z_thresh = self._ag_dict['zthresh']
        reopen = 'reopen' in self._ag_dict

        touch_test = np.abs(self._last_obs['state'][0]) < 0.97
        target, self._gripper_closed = autograsp_dynamics(self._previous_target_qpos, action,
                                                          self._gripper_closed, norm_gripper_z, z_thresh, reopen,
                                                          touch_test or self._prev_touch)
        self._prev_touch = touch_test
        return target