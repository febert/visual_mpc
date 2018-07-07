from python_visual_mpc.visual_mpc_core.envs.mujoco_env import BaseMujocoEnv
import python_visual_mpc
import cv2

BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/mjc_models/sawyer_assets/sawyer_xyz/'

class BaseSawyerEnv(BaseMujocoEnv):
    def __init__(self, throw):
        super().__init__(asset_base_path + 'sawyer_push_puck.xml')
        dual_render = self.render()

        cv2.imwrite('test1.png', dual_render[0])
        cv2.imwrite('test1.png', dual_render[1])
        raise NotImplementedError