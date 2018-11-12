from .policy import Policy
import copy
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.visual_mpc_rospkg.src.misc.camera_calib.calibrated_camera import CalibratedCamera
import numpy as np


class HardcodedPickPlace(Policy):
    def __init__(self, agent_params, policyparams, gpu_id, ngpu):
        assert agent_params['adim'] == 4, "Action dimension should be 4 for this policy!"
        self._robot_name = copy.deepcopy(agent_params['robot_name'])
        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self._calib_cam = CalibratedCamera(self._robot_name, self._hp.camera_name)

        self._pick_pos, self._drop_pos, self._phase, self._ctr = None, None, None, None

    def _default_hparams(self):
        default_dict = {
            'camera_name': 'kinect',
            'max_norm': 1.5,
            'eps': 1e-2
        }
        parent_params = super(HardcodedPickPlace, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t, state, desig_pix, goal_pix, high_bound, low_bound):
        if t == 0:
            norm = lambda x: np.clip((x[0] - low_bound[0,:3]) / (high_bound[0,:3] - low_bound[0,:3]), 0, 1)
            self._pick_pos = norm(self._calib_cam.camera_to_robot([desig_pix[0][0]]))
            self._drop_pos = norm(self._calib_cam.camera_to_robot([goal_pix[0][0]]))
            self._phase = 0
        action = np.zeros(4)

        if self._phase == 0:
            action[:2] = self._pick_pos[:2] - state[-1,:2]
            action[2] = 1
            act_norm = np.linalg.norm(action[:2])
            if act_norm > self._hp.max_norm:
                action[:2] *= self._hp.max_norm / act_norm
            if act_norm < self._hp.eps:
                self._phase += 1
                self._ctr = 0

        elif self._phase == 1:
            if self._ctr > 1:
                self._phase, self._ctr = 2, 0
            action[2] = -1./3
            self._ctr += 1

        elif self._phase == 2:
            if self._ctr > 1:
                self._phase, self._ctr = 3, 0
            action[2] = 1./3
            self._ctr += 1

        elif self._phase == 3:
            action[:2] = self._drop_pos[:2] - state[-1, :2]
            action[2] = 1
            act_norm = np.linalg.norm(action[:2])
            if act_norm > self._hp.max_norm:
                action[:2] *= self._hp.max_norm / act_norm
            if act_norm < self._hp.eps:
                self._phase += 1
                self._ctr = 0

        elif self._phase == 4:
            if self._ctr > 1:
                self._phase, self._ctr = 5, 0
            action[2] = -1./3
        else:
            raise NotImplementedError("Phase not implemented")

        return {'actions': action}