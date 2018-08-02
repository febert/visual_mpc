from .random_policy import RandomPickPolicy, Randompolicy
from .policy import Policy


def dereference_policy(name):
    mapping = {
        'randompolicy': 'Randompolicy',
        'randompickpolicy': 'RandomPickPolicy',
        'cem_goalimage_sawyer': 'CEM_controller'
    }
    name = mapping.get(name.lower(), name)
    policy_class = globals().get(name)

    if policy_class is None or not issubclass(policy_class, Policy):
        raise ValueError('Invalid policy name %s' % name)
    return policy_class
