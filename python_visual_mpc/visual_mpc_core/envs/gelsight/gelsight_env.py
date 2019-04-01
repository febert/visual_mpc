from python_visual_mpc.visual_mpc_core.envs.base_env import BaseEnv
import numpy as np
import random
from .testbench_control import TestBench
import time
import cv2
import deepdish as dd
import pickle
from enum import Enum
import serial


class TBConfig(Enum):

    BEARING = 0
    ANALOG_STICK = 1
    DICE = 2


class GelsightEnv(BaseEnv):

    """
    Environment class takes actions in real life using TestBench API and provides
    real world observations (images) back
    """

    # Coordinate axis bounds for Testbench
    MAX_X = 6000
    MAX_Y = 12000
    MAX_Z = 1000

    FLIPPED_X_TASKS = [TBConfig.ANALOG_STICK, TBConfig.DICE]

    statistics = {}
    with open('/home/stian/Documents/visual_mpc_dice/visual_mpc/python_visual_mpc/visual_mpc_core/envs/gelsight/bearing_stats.pkl', 'rb') as f:
        bearing_stats = pickle.load(f)
        statistics[TBConfig.BEARING] = (bearing_stats['mean'], bearing_stats['std'])

    with open('/home/stian/Documents/visual_mpc_dice/visual_mpc/python_visual_mpc/visual_mpc_core/envs/gelsight/analog_stick_stats.pkl', 'rb') as f:
        analog_stick_stats = pickle.load(f)
        statistics[TBConfig.ANALOG_STICK] = (analog_stick_stats['mean'], analog_stick_stats['std'])

    with open('/home/stian/Documents/visual_mpc_dice/visual_mpc/python_visual_mpc/visual_mpc_core/envs/gelsight/dice_2-14_stats.pkl', 'rb') as f:
        dice = pickle.load(f)
        statistics[TBConfig.DICE] = (dice['mean'], dice['std'])

    force_threshold = 20

    def __init__(self, env_params, res_state = None):
        self._hp = self._default_hparams()
        for name, value in env_params.items():
            print('setting param {} to value {}'.format(name, value))
            self._hp.set_hparam(name, value)
        print('Creating testbench environment... ')

        if not hasattr('side_cam_number', self._hp):
            self._hp.side_cam_number = None

        # Load in random initialization offsets from offset file, for reproducibility across
        # comparisons with baselines.
        if hasattr('offsets_path', self._hp):
            self.offsets = dd.io.load(self._hp.offsets_path)
        else:
            self.offsets = [(0, 0, 0)]

        self.tb = TestBench(self._hp.serial_port, self._hp.cam_number, self._hp.side_cam_number)
        self.task = self._hp.task
        # For dice, extra resetter mechanism
        if self.is_dice_task():
            self.resetter = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1) 
        self.mean, self.std = self.statistics[self.task]
        self.config_image_path = self._hp.config_image_path

        self.dice_servo_reset = self._hp.servo_reset
        self.dice_servo_loosen = self._hp.servo_loosen
        self.resets = 0

        self._setup_robot()

        self._base_adim, self._base_sdim = 3, 3
        self._adim, self._sdim, self.mode_rel = 3, 3, (True, True, True)

    def _default_hparams(self):
        default_dict = {'serial_port': '/dev/ttyACM0',
                        'cam_number': 0,
                        'side_cam_number': 1,
                        'robot_name': 'gelsight',
                        'ncam': 1,
                        'home_pos': (2650, 6040, 775),
                        'start_at_neutral': False,
                        'task': TBConfig.ANALOG_STICK}

        parent_params = BaseEnv._default_hparams(self)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _byte_encode(self, msg):
        # Encode message in utf-8 for serial messaging
        return bytes(msg, encoding='utf-8')

    def _reset_dice(self):
        # Reset (tighten) string on automatic dice reset setup
        self.resetter.write(self.byte_encode(str(self.dice_servo_reset) + '\n'))

    def _loosen_dice(self):
        # Loosen (allow dice to roll freely) string on automatic dice reset setup
        self.resetter.write(self.byte_encode(str(self.dice_servo_loosen) + '\n'))

    def _setup_robot(self):
        while not self.tb.ready():
            time.sleep(0.1)
            self.tb.update()
        if self.is_flipped_xreset():
            self.tb.flip_x_reset()
        if self.is_dice_task():
            self._reset_dice()
        self.reset()

    def clip_target(self, x, y, z):
        """
        Clip the target X, Y, Z coordinates based on the "bounding box" of MAX_X, MAX_Y, and MAX_Z (specified above)
        """
        x = min(self.MAX_X, max(0, x))
        y = min(self.MAX_Y, max(0, y))
        z = min(self.MAX_Z, max(0, z))
        return x, y, z

    def normalized_action_to_raw(self, action):
        return [action[0] * self.std['x_act'], action[1] * self.std['y_act'], action[2] * self.std['z_act']]

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

        # 0 action does nothing, grab image from testbench and return
        if all([a == 0 for a in action]):
            return self._get_obs()

        # Denormalize action so it makes sense in real life
        raw_action = self.normalized_action_to_raw(action)

        curr_state = self.tb.req_data()
        x, y, z = curr_state['x'], curr_state['y'], curr_state['z']
        target_x, target_y, target_z = self.clip_target(x + raw_action[0], y + raw_action[1], z + raw_action[2])
        forces = [curr_state['force_1'], curr_state['force_2'], curr_state['force_3'], curr_state['force_4']]
        mean_force = np.mean(forces)

        # If under force threshold, end effector can go vertically down. It can always go up.
        if mean_force < self.force_threshold or raw_action[2] < 0:
            self.tb.target_pos(target_x, target_y, target_z)
        else:
            print("Action rejected by force threshold")
            print('Current position: ')
            print(x, y, z)
            print('Forces: ')
            print(forces)

        # Wait for action to be taken on real robot
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
        images = np.array([img])
        side = self.tb.get_side_cam_frame()
        return {'images': images,
                'state': xyz_state,
                'side_img': side}

    def sleep(self, secs):
        t_end = time.time() + secs
        while time.time() < t_end:
            self.tb.update()

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

        if self.is_dice_task():
            self.perform_interactive_reset(output_path=self.config_image_path)

        off = self.offsets[self.resets % len(self.offsets)] 
        print('Current offset: {}'.format(off))
        new_pos = (self._hp.home_pos[0] + off[0], self._hp.home_pos[1] + off[1], self._hp.home_pos[2] + off[2]) 
        self.tb.target_pos(*new_pos)
        while self.tb.busy():
            self.tb.update()
        self.resets = (self.resets + 1) % 100
        reset_state = []
        return self._get_obs(), reset_state

    def perform_interactive_reset(self, output_path):
        """
        The 'interactive reset' allows you to do initial position setup for specific tasks.
        """

        off = self.offsets[self.resets % len(self.offsets)]
        pre_reset = None
        while pre_reset != 'b':
            pre_reset = input('please confirm you have viewed the top of the dice by pressing b:')
        self._reset_dice()
        self.sleep(1)
        self._loosen_dice()

        self.sleep(1)
        do_reset = input('Perform dice positioning setup? y/n: ')
        if do_reset != 'n':
            confirm = None
            calib_pos = (self._hp.home_pos[0] + off[0], self._hp.home_pos[1] + off[1], self._hp.home_pos[2] + off[2])
            while confirm != 'conf':
                self.tb.reset_z()
                while self.tb.busy():
                    self.tb.update()
                post_set = None
                while post_set != 'y':
                    post_set = input('Finished moving? y/n: ')
                self.tb.target_pos(*calib_pos)
                while self.tb.busy():
                    self.tb.update()
                self.sleep(0.2)
                cv2.imwrite(output_path, self.tb.get_frame())
                confirm = input(
                    'please confirm initial position by typing conf, if you need to keep adjusting, anything else: ')
        self._reset_dice()
        self.sleep(1)

    def is_dice_task(self):
        return self.task == TBConfig.DICE

    def is_flipped_xreset(self):
        return self.task in self.FLIPPED_X_TASKS

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


