from python_visual_mpc.visual_mpc_core.envs.sawyer_sim.autograsp_env import AutograspSawyerEnv


class AGOpenActionEnv(AutograspSawyerEnv):
    def __init__(self, env_params):
        super().__init__(env_params)
        self._adim = 5

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim

        target = super()._next_qpos(action[:-1])
        if action[-1] <= 0:                     #if policy outputs an "open" action then override auto-grasp
            self._gripper_closed = False
            target[-1] = -1

        return target

