""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
from lsdc.utility.general_utils import check_shape


class Random_impedance_point(Policy):
    """
    Random Policy
    """
    def __init__(self, agent_params, policy_params):
        Policy.__init__(self)
        self.x_g = np.array([0,0])

    def act(self, x, xdot, sample_images, t, init_model= None):
        """
        Return a random action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            x_data_idx: data_files indexes for x
        """

        new_point_freq = 13

        if t % new_point_freq ==0:
            #set new target point
            self.x_g = np.random.uniform(-.4, .4, 2)

        # self.x_g = np.array([0.1,0.1])
        c = 40
        d = 20

        f = (self.x_g - x)*c -xdot*d

        return f