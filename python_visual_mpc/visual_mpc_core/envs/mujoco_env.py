import gym
from mujoco_py import load_model_from_path, MjSim
import numpy as np
from gym.utils import seeding

class BaseMujocoEnv(gym.Env):
    def __init__(self,  model_path, height=480, width=640):
        self._frame_height = height
        self._frame_width = width

        self._reset_sim(model_path)

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