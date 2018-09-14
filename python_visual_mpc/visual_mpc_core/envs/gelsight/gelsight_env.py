from python_visual_mpc.visual_mpc_core.envs.base_env import BaseEnv
import numpy as np
import random
import os
from .testbench_control import TestBench
import time
import pdb
import matplotlib.pyplot as plt
import cv2
import deepdish as dd
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_cost_functions import mse_based_cost

class GelsightEnv(BaseEnv):

    MAX_X = 6000
    MAX_Y = 12000
    MAX_Z = 1150
    mean = {
        'force_1': 5.548266944444444,
        'z': 1115.4815277777777,
        'x_act': 0,
        'force_4': 5.346665555555556,
        'y_act': 0,
        'y': 6045.260583333334,
        'force_3': 6.150440555555555,
        'force_2': 6.4838152777777776,
        'z_act': 0,
        'x': 2644.52925
    }
    std = {
        'force_1': 8.618291543401973,
        'z': 34.865522493962516,
        'x_act': 40.55583610822862,
        'force_4': 5.871973470396116,
        'y_act': 40.9047147811397,
        'y': 212.2458477847846,
        'force_3': 7.239953607917641,
        'force_2': 4.568602618451527,
        'z_act': 6.070706704238715,
        'x': 209.59929224857643

    }
    GOAL_DIR = 'goals/'
    force_threshold = 15

    offsets = dd.io.load('/home/stephentian/Documents/visual_mpc/python_visual_mpc/visual_mpc_core/envs/gelsight/offset150.hd5')

    def __init__(self, env_params, res_state = None):
        self._hp = self._default_hparams()
        for name, value in env_params.items():
            print('setting param {} to value {}'.format(name, value))
            self._hp.set_hparam(name, value)
        print('Creating testbench environment... ')
        self.tb = TestBench(self._hp.serial_port, self._hp.cam_number)
        self.resets = 0

        self.i = 0
        self._setup_robot()

        self._base_adim, self._base_sdim = 3, 3
        self._adim, self._sdim, self.mode_rel = 3, 3, (True, True, True)

    def _default_hparams(self):
        default_dict = {'serial_port': '/dev/ttyACM0',
                        'cam_number': 0,
                        'robot_name': 'gelsight',
                        'ncam': 1,
                        'home_pos': (2650, 6040, 775)}

        parent_params = BaseEnv._default_hparams(self)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _setup_robot(self):
        while not self.tb.ready():
            time.sleep(0.1)
            self.tb.update()
        self.reset()

    def clip_target(self, x, y, z):
        x = min(self.MAX_X, max(0, x))
        y = min(self.MAX_Y, max(0, y))
        z = min(self.MAX_Z, max(0, z))
        return x, y, z

    def step(self, action):
        """
        Applies the action and steps simulation
        :param action: action at time-step
        :return: obs dict where:
                  -each key is an observation at that step
                  -keys are constant across entire datastep (e.x. every-timestep has 'state' key)
                  -keys corresponding to numpy arrays should have constant shape every timestep (for caching)
                  -images should be placed in the 'images' key in a (ncam, ...) array
        """
        if all([a == 0 for a in action]):
            return self._get_obs()

        # Denormalize action so it makes sense in real life
        raw_action = [action[0] * self.std['x_act'], action[1] * self.std['y_act'], action[2] * self.std['z_act']]

        curr_state = self.tb.req_data()
        x, y, z = curr_state['x'], curr_state['y'], curr_state['z']
        target_x, target_y, target_z = self.clip_target(x + raw_action[0], y + raw_action[1], z + raw_action[2])
        forces = [curr_state['force_1'], curr_state['force_2'], curr_state['force_3'], curr_state['force_4']]
        mean_force = np.mean(forces)
        if mean_force < self.force_threshold:
            self.tb.target_pos(target_x, target_y, target_z)
        else:
            print("Action rejected by force threshold")

        while self.tb.busy():
            self.tb.update()

        obs = self._get_obs()
        return obs

    def normalize(self, data):
        for feat in data:
            if feat != 'slip':
                data[feat] = (data[feat] - self.mean[feat]) / self.std[feat]
        return data

    def _get_obs(self):
        state = self.tb.req_data()
        normalized_state = self.normalize(state)
        xyz_state = np.array([normalized_state['x'],
                              normalized_state['y'],
                              normalized_state['z']])

        img = self.tb.get_frame()
        # img = img[:,:,::-1]
        images = np.array([img])
        cv2.imwrite('/home/stephentian/Documents/visual_mpc/experiments/im_{}.jpg'.format(self.i), img)
        self.i += 1
        return {'images': images,
                'state': xyz_state}

    def reset(self):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        if not self.resets % 100:
            print("Recalibrating axes.")
            self.tb.reset()
            while self.tb.busy():
                self.tb.update()
        self.tb.reset_z()
        while self.tb.busy():
            self.tb.update()
        off = self.offsets[self.resets % 30]
        new_pos = (self._hp.home_pos[0] - 150, self._hp.home_pos[1], self._hp.home_pos[2]) # TODO: Change back to offset dict
        self.tb.target_pos(*new_pos)
        while self.tb.busy():
            self.tb.update()
        self.resets = (self.resets + 1) % 100
        reset_state = []
        return self._get_obs(), reset_state

    def valid_rollout(self):
        """
        Checks if the environment is currently in a valid state
        Common invalid states include:
            - object falling out of bin
            - mujoco error during rollout
        :return: bool value that is False if rollout isn't valid
        """
        return True

    def get_obj_desig_goal(self, save_dir, collect_goal_image=False, ntasks=1):
        pass

    def set_goal_obj_pose(self, pose):
        pass

    def eval(self, target_width, save_dir, ntasks, goal_image):
        pass


    @property
    def adim(self):
        """
        :return: Environment's action dimension
        """
        return self._adim

    @property
    def sdim(self):
        """
        :return: Environment's state dimension
        """
        return self._sdim

    @property
    def ncam(self):
        """
        Gelsight environment has ncam cameras
        """
        return 1

    @property
    def num_objects(self):
        """
        :return: Dummy value for num_objects (used in general_agent logic)
        """
        return 0

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

