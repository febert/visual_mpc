""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
from lsdc.utility.general_utils import check_shape


class Randompolicy(Policy):
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
        force_magnitude = 2
        return np.random.uniform(-force_magnitude,force_magnitude,2)
