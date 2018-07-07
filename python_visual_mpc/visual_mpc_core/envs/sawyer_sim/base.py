from python_visual_mpc.visual_mpc_core.envs.mujoco_env import BaseMujocoEnv
import python_visual_mpc
import cv2
import numpy as np
BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/mjc_models/sawyer_assets/sawyer_xyz/'

class BaseSawyerEnv(BaseMujocoEnv):
    def __init__(self, throw):
        super().__init__(asset_base_path + 'sawyer_pick_and_place.xml')
        dual_render = self.render()

        cv2.imwrite('test1.png', dual_render[0, :,:,::-1])
        cv2.imwrite('test2.png', dual_render[1, : ,: ,::-1])

        self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
        self.sim.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

        for i in range(100):
            self.sim.step()
        
        raise NotImplementedError