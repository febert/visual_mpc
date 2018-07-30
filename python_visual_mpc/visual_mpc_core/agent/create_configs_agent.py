""" This file defines an agent for the MuJoCo simulator environment. """
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import copy
import numpy as np
import pickle
from PIL import Image
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
import os
import cv2
from python_visual_mpc.visual_mpc_core.algorithm.policy import get_policy_args


from .general_agent import GeneralAgent


class CreateConfigAgent(GeneralAgent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def rollout(self, policy, i_tr):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_tr: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        # Take the sample.
        self.large_images_traj, self.traj_points= [], None
        obs = self._post_process_obs(self.env.reset(), True)
        agent_data, policy_outputs = {}, []
        agent_data['traj_ok'] = True

        for t in range(self._hyperparams['T']):
            self.env.move_arm()
            self.env.move_objects()
            try:
                obs = self._post_process_obs(self.env._get_obs(None))
            except ValueError:
                return {'traj_ok': False}, None, None

        return agent_data, obs, policy_outputs

