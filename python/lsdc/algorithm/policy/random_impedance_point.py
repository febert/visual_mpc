""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
from lsdc.utility.general_utils import check_shape


class Random_impedance_point(Policy):
    """
    Random Policy
    """
    def __init__(self):
        Policy.__init__(self)



    def act(self, x, obs, t, noise=None):
        """
        Return a random action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # import pdb; pdb.set_trace()
        new_point_freq = 20

        if t % new_point_freq ==0:
            #set new target point
            x_g = np.random.uniform(-.4, .4, 2)

        return 0