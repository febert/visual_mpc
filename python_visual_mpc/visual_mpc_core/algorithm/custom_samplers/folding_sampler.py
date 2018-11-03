import copy
import numpy as np


class FoldingSampler:
    def __init__(self, sigma, mean, hp, repeat, base_adim):
        self._hp = hp
        naction_steps = hp.nactions
        assert base_adim == 4, "Requires base action dimension of 4"
        assert naction_steps >= 5, "Requires at least 5 steps"
        self._base_mean = mean
        self._full_sigma = copy.deepcopy(sigma)
        self._base_sigma = self._full_sigma[:4, :4]
        self._adim = base_adim
        self._repeat = repeat
        self._steps = naction_steps

    def sample(self, M, current_state):
        assert M % 3 == 0, "splits samples into setting with 3 means"
        ret_actions = np.zeros((M, self._steps, self._adim))
        per_split, current_state = M / 3, current_state[-1, :2]

        lower_sigma = copy.deepcopy(self._base_sigma)
        lower_sigma[:2, :2] /= 10
        lower_sigma[3, 3] /= 2

        for i in range(per_split):
            first_pnt, second_pnt = np.random.uniform(size=2), np.random.uniform(size=2)

            delta_first, delta_second = (first_pnt - current_state) / self._repeat, \
                                        (second_pnt - first_pnt) / self._repeat

            mean = np.array([delta_first[0], delta_first[1], 1, 0.])
            ret_actions[i, 0] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 1] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([0, 0., 1, 0])
            ret_actions[i, 2] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([delta_second[0], delta_second[1], 1, 0])
            ret_actions[i, 3] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 4] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            if self._steps > 5:
                ret_actions[i, self._steps:] = np.random.multivariate_normal(np.zeros(4), self._base_sigma,
                                                                             self._steps - 5)

        for i in range(per_split, 2 * per_split):
            second_pnt = np.random.uniform(size=2)

            delta_second = (second_pnt - current_state) / self._repeat

            mean = np.array([0, 0, 1, 0.])
            ret_actions[i, 0] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([delta_second[0], delta_second[1], 1, 0])
            ret_actions[i, 1] = np.random.multivariate_normal(mean, self._base_sigma, 1).reshape(-1)

            mean = np.array([0, 0., -1, 0])
            ret_actions[i, 2] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            mean = np.array([0, 0., 0, 0])
            ret_actions[i, 3:] = np.random.multivariate_normal(mean, lower_sigma, 1).reshape(-1)

            if self._steps > 5:
                ret_actions[i, self._steps:] = np.random.multivariate_normal(np.zeros(4), self._base_sigma,
                                                                             self._steps - 5)
        default_actions = np.random.multivariate_normal(self._base_mean, self._full_sigma, per_split)
        ret_actions[2 * per_split:] = default_actions.reshape((per_split, self._steps, self._adim))

        ret_actions[:, :, :3] = np.clip(ret_actions[:, :, :3], -np.array(self._hp.max_shift),
                                        np.array(self._hp.max_shift))

        return np.repeat(ret_actions, self._repeat, axis=1)

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
            'max_shift': [1. / 5, 1. / 5, 1. / 3]
        }
        return hparams_dict

    def _initial_samples(self):
        return