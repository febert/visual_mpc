from python_visual_mpc.visual_mpc_core.envs.mujoco_env.sawyer_sim.autograsp_sawyer_mujoco_env import AutograspSawyerMujocoEnv
import numpy as np


class AGOpenActionEnv(AutograspSawyerMujocoEnv):
    def __init__(self, env_params, reset_state=None):
        super().__init__(env_params, reset_state)
        self._adim, self._threshold = 5, self._hp.open_action_threshold

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('open_action_threshold', 0.)
        return parent_params

    def _next_qpos(self, action):
        assert action.shape[0] == 5
        target = super()._next_qpos(action[:-1])
        if action[-1] <= self._threshold:                     #if policy outputs an "open" action then override auto-grasp
            self._gripper_closed = False
            target[-1] = -1

        return target

