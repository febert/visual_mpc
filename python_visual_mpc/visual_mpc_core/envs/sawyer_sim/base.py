from python_visual_mpc.visual_mpc_core.envs.mujoco_env import BaseMujocoEnv
import python_visual_mpc
import cv2
import numpy as np
import mujoco_py
from pyquaternion import Quaternion
import moviepy.editor as mpy

def quat_to_zangle(quat):
    angle = (Quaternion(axis = [0,1,0], angle = np.pi/2).inverse * Quaternion(quat)).angle
    if angle < 0:
        return 2 * np.pi + angle
    return angle

def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, -1], angle= zangle)).elements

BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/mjc_models/sawyer_assets/sawyer_xyz/'

class BaseSawyerEnv(BaseMujocoEnv):
    def __init__(self, throw):
        super().__init__(asset_base_path + 'sawyer_reach.xml')
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )
        clip0, clip1 = [],[]

        for i, zangle in enumerate(np.linspace(0, 2 * np.pi, 100)):
            print(self.sim.data.ctrl.shape)
            if i % 20 < 10:
                self.sim.data.ctrl[0] = 1
                self.sim.data.ctrl[1] = -1
            else:
                self.sim.data.ctrl[0] = -1
                self.sim.data.ctrl[1] = 1
            print(self.sim.data.ctrl)
            self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.2]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(zangle))
            print('target', zangle_to_quat(zangle))
            print('real', zangle, 'inv', quat_to_zangle(zangle_to_quat(zangle)))

            for _ in range(100):
                self.sim.step()
            print('xyz', self.sim.data.get_body_xpos('hand').copy())
            print('quat', self.sim.data.get_body_xquat('hand').copy())
            print('joints', self.sim.data.qpos)
            dual_render = self.render()

            clip0.append(dual_render[0])
            clip1.append(dual_render[1])
        for c, name in zip([clip0, clip1], ['test0.gif', 'test1.gif']):
            clip = mpy.ImageSequenceClip(c, fps = 10)
            clip.write_gif(name)

        raise NotImplementedError
