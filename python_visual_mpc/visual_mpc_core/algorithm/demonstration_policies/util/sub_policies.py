from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import numpy as np
from tensorflow.contrib.training import HParams


class Wiggle(Policy):
    def __init__(self,  ag_params, policyparams, gpu_id, ngpu):
        Policy.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._params = self.get_default_hparams().override_from_dict(policyparams)

        noise_params = (self._params.x_noise, self._params.y_noise, self._params.z_noise, self._params.theta_noise)
        self._gaus_noise_mean = np.array([x[0] for x in noise_params] + [0])
        self._gaus_noise_std = np.diag([x[1] for x in noise_params] + [0])

    def get_default_hparams(self):
        default_dict = {
            'x_noise': (0, 0.1),  # Gaussian (mean, std)
            'y_noise': (0, 0.1),  # Gaussian (mean, std)
            'z_noise': (0, 0.15),  # Gaussian (mean, std)
            'theta_noise': (0, np.pi/3),  # Gaussian (mean, std)
            'gripper_noise': 0.5,     # probability that gripper opens
            'truncate_action': np.array([0.1, 0.1, 0.15, np.pi / 4, 1])
        }
        return HParams(**default_dict)

    def act(self):
        noise_t = self._gaus_noise_std.dot(np.random.normal(size=(5, 1))).reshape(-1) + self._gaus_noise_mean
        if np.random.uniform() < self._params.gripper_noise:
            noise_t[-1] = 1
        else:
            noise_t[-1] = -1
        action = np.clip(noise_t, -self._params.truncate_action, self._params.truncate_action)

        return {'actions': action}


class WiggleToObject(Policy):
    def __init__(self,  ag_params, policyparams, gpu_id, ngpu):
        Policy.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._policy_started, self._last_t = False, -1
        self._params = self.get_default_hparams().override_from_dict(policyparams)

    def get_default_hparams(self):
        default_dict = {
            'x_noise': (0, 0.0001),       # Gaussian (mean, std)
            'y_noise': (0, 0.0001),       # Gaussian (mean, std)
            'z_noise': (0, 0.00015),       # Gaussian (mean, std)
            'theta_noise': (0, np.pi/8000),   # Gaussian (mean, std)
            'finger_noise': 0.,        # probability that gripper randomly closes during a timestep
            'max_norm': 0.15,
            'max_rot': 2 * np.pi,
            'truncate_rot': np.pi / 4,
            'rand_cutoff': 6,
            'tolerance': 0.04,
            'xyz_bias': [0, 0, 0.15]
        }
        return HParams(**default_dict)

    def _make_action(self, t, target_xyz, current_xyz, current_theta):
        noise_t = self._gaus_noise_std.dot(np.random.normal(size=(5, 1))).reshape(-1) + self._gaus_noise_mean
        delta_t = np.zeros(5)

        delta_t[:3] = target_xyz - current_xyz + self._params.xyz_bias
        delta_t[3] = np.clip(self._target_rot - current_theta, -self._params.truncate_rot, self._params.truncate_rot)

        if self._params.rand_cutoff < 0 or t < self._last_t:
            delta_t += noise_t

        delta_norm = np.linalg.norm(delta_t[:3])
        if delta_norm > self._params.max_norm:
            delta_t[:3] *= self._params.max_norm / delta_norm

        if np.random.uniform() < self._params.finger_noise:
            delta_t[-1] = 1
        else:
            delta_t[-1] = -1

        return delta_t

    def act(self, t, state, object_poses_full):
        if not self._policy_started:
            if self._params.rand_cutoff >= 0:
                self._last_t = t + self._params.rand_cutoff
            noise_params = (self._params.x_noise, self._params.y_noise, self._params.z_noise, self._params.theta_noise)
            self._gaus_noise_mean = np.array([x[0] for x in noise_params] + [0])
            self._gaus_noise_std = np.diag([x[1] for x in noise_params] + [0])

            n_objects = object_poses_full.shape[1]
            self._target_object = np.random.randint(n_objects)
            self._target_rot = np.random.uniform(0, self._params.max_rot)

            self._policy_started = True

        action = self._make_action(t, object_poses_full[-1, self._target_object, :3], state[-1, :3], state[-1, 3])
        return {'actions': action}

    def is_done(self, state, object_poses_full):
        if self._policy_started:
            return np.linalg.norm(object_poses_full[-1, self._target_object, :2] - state[-1, :2]) < self._params.tolerance
        return False

class WiggleToXYZ(WiggleToObject):
    def get_default_hparams(self):
        parent_params = super().get_default_hparams()
        parent_params.add_hparam('bounds', [[-0.17, 0.62, 0.2], [0.17, 0.85, 0.3]])
        return parent_params

    def act(self, t, state):
        if not self._policy_started:
            if self._params.rand_cutoff >= 0:
                self._last_t = t + self._params.rand_cutoff
            noise_params = (self._params.x_noise, self._params.y_noise, self._params.z_noise, self._params.theta_noise)
            self._gaus_noise_mean = np.array([x[0] for x in noise_params] + [0])
            self._gaus_noise_std = np.diag([x[1] for x in noise_params] + [0])

            self._target_xyz = np.random.uniform(self._params.bounds[0], self._params.bounds[1])
            self._target_rot = np.random.uniform(0, self._params.max_rot)

            self._policy_started = True
        action = self._make_action(t, self._target_xyz, state[-1, :3], state[-1, 3])
        return {'actions': action}

    def is_done(self, state):
        if self._policy_started:
            return np.linalg.norm(self._target_xyz[:2] - state[-1, :2]) < self._params.tolerance
        return False

class WiggleAndLift(Policy):
    def __init__(self,  ag_params, policyparams, gpu_id, ngpu):
        Policy.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._ctr = 0
        self._params = self.get_default_hparams().override_from_dict(policyparams)

        noise_params = (self._params.x_noise, self._params.y_noise, self._params.z_noise, self._params.theta_noise)
        self._gaus_noise_mean = np.array([x[0] for x in noise_params] + [0])
        self._gaus_noise_std = np.diag([x[1] for x in noise_params] + [0])

    def get_default_hparams(self):
        default_dict = {
            'x_noise': (0, 0.0001),       # Gaussian (mean, std)
            'y_noise': (0, 0.0001),       # Gaussian (mean, std)
            'z_noise': (0, 0.00001),       # Gaussian (mean, std)
            'theta_noise': (0, 0.0001),   # Gaussian (mean, std)
            'truncate_action': np.array([0.01, 0.01, 0.08, np.pi / 8, 1]),
            'target_z': 0.3
        }
        return HParams(**default_dict)

    def act(self, state):
        noise_t = self._gaus_noise_std.dot(np.random.normal(size=(5, 1))).reshape(-1) + self._gaus_noise_mean
        delta_t = np.zeros(5)

        if self._ctr % 5 < 2:
            delta_t[2], delta_t[4] = -1, -1
        elif self._ctr % 5 == 2:
            delta_t[4] = 1
        else:
            delta_t[2], delta_t[4] = 1, 1

        self._ctr += 1
        action = np.clip(delta_t + noise_t, -self._params.truncate_action, self._params.truncate_action)
        return {'actions': action}

    @property
    def ctr(self):
        return self._ctr


class WiggleAndPlace(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        Policy.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._ctr = 0
        self._params = self.get_default_hparams().override_from_dict(policyparams)

        noise_params = (self._params.x_noise, self._params.y_noise, self._params.z_noise, self._params.theta_noise)
        self._gaus_noise_mean = np.array([x[0] for x in noise_params] + [0])
        self._gaus_noise_std = np.diag([x[1] for x in noise_params] + [0])

    def get_default_hparams(self):
        default_dict = {
            'x_noise': (0, 0.1),  # Gaussian (mean, std)
            'y_noise': (0, 0.1),  # Gaussian (mean, std)
            'z_noise': (0, 0.1),  # Gaussian (mean, std)
            'theta_noise': (0, np.pi/8),  # Gaussian (mean, std)
            'truncate_action': np.array([0.01, 0.01, 0.2, np.pi / 8, 1]),
            'target_z': 0.3
        }
        return HParams(**default_dict)

    def act(self, state):
        noise_t = self._gaus_noise_std.dot(np.random.normal(size=(5, 1))).reshape(-1) + self._gaus_noise_mean
        delta_t = np.zeros(5)

        if self._ctr % 5 < 2:
            delta_t[2], delta_t[4] = -1, 1
        elif self._ctr % 5 == 2:
            delta_t[4] = -1
        else:
            delta_t[2], delta_t[4] = 1, -1

        self._ctr += 1
        action = np.clip(delta_t + noise_t, -self._params.truncate_action, self._params.truncate_action)
        return {'actions': action}

    @property
    def ctr(self):
        return self._ctr