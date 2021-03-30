from gym.envs.registration import register
from gym_cellular_automata.operator import Operator

RESGISTERED_CA_ENVS = ("forest-fire-v0", "forest-fire-v1", "forest-fire-v1")

register(
    id=RESGISTERED_CA_ENVS[0],
    entry_point="gym_cellular_automata.envs.forest_fire.v0:ForestFireEnv",
)


register(
    id=RESGISTERED_CA_ENVS[1],
    entry_point="gym_cellular_automata.envs.forest_fire.v1:ForestFireEnv",
)


"""
register(
    id=RESGISTERED_CA_ENVS[2],
    entry_point="gym_cellular_automata.envs.forest_fire.v2:ForestFireEnv",
)
"""

__all__ = ["Operator", "RESGISTERED_CA_ENVS"]
