""" This file defines the base class for the policy. """
import abc
import abc, six
from funcsigs import signature, Parameter


def get_policy_args(policy, obs, t):
    policy_args = {}
    policy_signature = signature(policy.act)  # Gets arguments required by policy
    for arg in policy_signature.parameters:  # Fills out arguments according to their keyword
        value = policy_signature.parameters[arg].default
        if arg in obs:
            value = obs[arg]
        elif arg == 't':
            value = t

        if value is Parameter.empty:
            # required parameters MUST be set by agent
            raise ValueError("Required Policy Param {} not set in agent".format(arg))
        policy_args[arg] = value
    return policy_args


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
