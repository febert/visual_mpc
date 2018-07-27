""" This file defines the base class for the policy. """
import abc, six
from funcsigs import signature, Parameter


def get_policy_args(policy, obs, t, i_tr):
    policy_args = {}
    policy_signature = signature(policy.act)  # Gets arguments required by policy
    for arg in policy_signature.parameters:  # Fills out arguments according to their keyword
        value = policy_signature.parameters[arg].default
        if arg in obs:
            value = obs[arg]
        elif arg == 't':
            value = t
        elif arg == 'i_tr':
            value = i_tr

        if value is Parameter.empty:
            # required parameters MUST be set by agent
            raise ValueError("Required Policy Param {} not set in agent".format(arg))
        policy_args[arg] = value

    return policy_args


@six.add_metaclass(abc.ABCMeta)
class Policy(object):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """ Computes actions from states/observations. """
        pass

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

    def reset(self):
        raise NotImplementedError


class DummyPolicy(object):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """ Computes actions from states/observations. """
        pass

    @abc.abstractmethod
    def act(self, *args):
        pass

    def reset(self):
        pass
