from .base_mujoco_env import BaseMujocoEnv


class BaseDemoEnv(BaseMujocoEnv):
    def reset(self):
        self._demo_t, self._cur_stage = 0, -1
        obs, reset = super().reset()
        obs = self.insert_stage(obs)
        return obs, reset

    def step(self, action):
        self._demo_t += 1
        return self.insert_stage(super().step(action))

    def insert_stage(self, obs_dict):
        if 'stage' in obs_dict:
            for k in obs_dict:
                print('key {}'.format(k))
                print('val {}'.format(obs_dict[k]))

        assert 'stage' not in obs_dict, "Demonstration Environment sets Stage"
        obs_dict['stage'] = self.get_stage()
        return obs_dict

    def get_stage(self):
        raise NotImplementedError

    def has_goal(self):
        return True

    def goal_reached(self):
        raise NotImplementedError