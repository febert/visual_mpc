""" This file defines the base class for the policy. """
import abc
import abc, six

@six.add_metaclass(abc.ABCMeta)
class Policy(object):
    """ Computes actions from states/observations. """

    @abc.abstractmethod
    def act(self, *args):
        """
        Args:
            Request necessary arguments in definition
            (see Agent code)
        Returns:
            A dict of outputs D
               -One key in D, 'actions' should have the action for this time-step
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def set_meta_data(self, meta):
        """
        Set meta data_files for policy (e.g., domain image, multi modal observation sizes)
        Args:
            meta: meta data_files.
        """
        return
