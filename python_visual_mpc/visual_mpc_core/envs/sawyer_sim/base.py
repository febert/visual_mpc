from python_visual_mpc.visual_mpc_core.envs.mujoco_env import BaseMujocoEnv
from python_visual_mpc.visual_mpc_core.envs.cartgripper_env.base_cartgripper import zangle_to_quat
import python_visual_mpc
import cv2
import numpy as np
import mujoco_py
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
        for i, zangle in enumerate(np.linspace(0, 2 * np.pi, 100)):
            self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(zangle))
            print('target', zangle_to_quat(zangle))
            for _ in range(100):
                self.sim.step()
            print('xyz', self.sim.data.get_body_xpos('hand').copy())
            print('quat', self.sim.data.get_body_xquat('hand').copy())
            print('joints', self.sim.data.qpos)
            dual_render = self.render()

            cv2.imwrite('test/test{}.png'.format(i), dual_render[0, :,:,::-1])
            print()
        raise NotImplementedError
