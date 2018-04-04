""" This file defines the base class for the policy. """
import abc


class Policy(object, metaclass=abc.ABCMeta):
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

    def set_meta_data(self, meta):
        """
        Set meta data_files for policy (e.g., domain image, multi modal observation sizes)
        Args:
            meta: meta data_files.
        """
        return
