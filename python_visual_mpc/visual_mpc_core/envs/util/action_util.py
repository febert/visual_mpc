from scipy.interpolate import CubicSpline
import numpy as np


class CSpline(object):
    def __init__(self, p_1, p_2):
        self.cs = CubicSpline(np.array([0.0, 1.0]), np.array([p_1, p_2]), bc_type="clamped")

    def get(self, t):
        t = np.array(t)
        return self.cs(t), self.cs(t, nu=1), self.cs(t, nu=2)


def autograsp_dynamics(prev_target_qpos, action, gripper_closed, gripper_zpos, zthresh, reopen, is_touching):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:4] = action[:4] + prev_target_qpos[:4]

    if gripper_zpos < zthresh:
        gripper_closed = True
    elif reopen and not is_touching:
        gripper_closed = False

    if gripper_closed:
        target_qpos[4] = 1
    else:
        target_qpos[4] = -1

    return target_qpos, gripper_closed
