from lsdc.algorithm.policy.random_policy import Randompolicy

policy = {
    'type' : Randompolicy,
    'initial_var': 10,
    'numactions': 8, # number of consecutive actions
    'repeats': 3, # number of repeats for each action
}

agent = {
    'random_baseline': True,
    'T': 24
}