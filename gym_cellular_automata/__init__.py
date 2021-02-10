import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))

from gym_cellular_automata.operator import Operator

from gym.envs.registration import register

sys.path.insert(1, directory)

RESGISTERED_CA_ENVS = 'forest-fire-v0',

register(
    id='forest-fire-v0',
    entry_point='gym_cellular_automata.envs:ForestFireEnv',
)

__all__ = ['Operator', 'RESGISTERED_CA_ENVS']
