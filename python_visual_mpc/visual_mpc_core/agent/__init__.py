from .agent_mjc import AgentMuJoCo
from .agent_fetch import AgentFetch


def dereference_agent(name):
    mapping = {
        'agentmujoco' : 'AgentMuJoCo',
        'agentfetch' : 'AgentFetch'
    }

    name = mapping.get(name.lower(), name)
    agent_class = globals().get(name)

    if agent_class is None:
        raise ValueError('Invalid agent name %s' % name)
    return agent_class
