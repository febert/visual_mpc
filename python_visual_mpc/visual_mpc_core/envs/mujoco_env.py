import gym
from mujoco_py import load_model_from_path, MjSim
import numpy as np
from gym.utils import seeding
from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point
import copy
class BaseMujocoEnv(gym.Env):
    def __init__(self,  model_path, height=480, width=640):
        self._frame_height = height
        self._frame_width = width

        self._reset_sim(model_path)

        self._base_adim, self._base_sdim = None, None                 #state/action dimension of Mujoco control
        self._adim, self._sdim = None, None   #state/action dimension presented to agent
        self.num_objects = None

    def _reset_sim(self, model_path):
        """
        Creates a MjSim from passed in model_path
        :param model_path: Absolute path to model file
        :return: None
        """
        self._model_path = model_path
        self.sim = MjSim(load_model_from_path(self._model_path))

    def render(self, mode='dual'):
        """ Renders the enviornment.
        Implements custom rendering support. If mode is:

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        cameras = ['maincam']
        if mode == 'dual':
            cameras = ['maincam', 'leftcam']
        elif mode == 'leftcam':
            cameras = ['leftcam']

        images = np.zeros((len(cameras), self._frame_height, self._frame_width, 3), dtype=np.uint8)
        for i, cam in enumerate(cameras):
            images[i] = self.sim.render(self._frame_width, self._frame_height, camera_name=cam)
        return images

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_desig_pix(self, ncam, target_width, round=True):
        qpos_dim = self._base_sdim      # the states contains pos and vel
        assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self.num_objects
        desig_pix = np.zeros([ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam in range(ncam):
            for i in range(self.num_objects):
                fullpose = self.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
                d = project_point(fullpose[:3], icam)
                d = np.stack(d) / ratio
                if round:
                    d = np.around(d).astype(np.int)
                desig_pix[icam, i] = d
        return desig_pix

    def get_goal_pix(self, ncam, target_width, goal_obj_pose, round=True):
        goal_pix = np.zeros([ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam in range(ncam):
            for i in range(self.num_objects):
                g = project_point(goal_obj_pose[i, :3], icam)
                g = np.stack(g) / ratio
                if round:
                    g= np.around(g).astype(np.int)
                goal_pix[icam, i] = g
        return goal_pix

    def snapshot_noarm(self):
        qpos = copy.deepcopy(self.sim.data.qpos)
        qpos[2] -= 10
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()
        image = self.render('maincam').squeeze()
        qpos[2] += 10
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()

        return image

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim
