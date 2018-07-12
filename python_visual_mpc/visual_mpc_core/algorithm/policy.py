""" This file defines the base class for the policy. """
import abc, six

@six.add_metaclass(abc.ABCMeta)
class Policy(object):
    """ Computes actions from states/observations. """

    @abc.abstractmethod
    def act(self,traj, t):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")
