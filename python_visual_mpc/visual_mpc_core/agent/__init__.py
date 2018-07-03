from .agent_mjc import AgentMuJoCo


def dereference_agent(name):
    mapping = {
        'agentmujoco' : 'AgentMuJoCo'
    }

    name = mapping.get(name.lower(), name)
    agent_class = globals().get(name)

    if agent_class is None:
        raise ValueError('Invalid agent name %s' % name)
    return agent_class
