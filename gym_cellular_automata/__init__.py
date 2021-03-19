from gym.envs.registration import register
from gym_cellular_automata.operator import Operator

RESGISTERED_CA_ENVS = ("forest-fire-v0", "forest-fire-v1")

register(id="forest-fire-v0", entry_point="gym_cellular_automata.envs:ForestFireEnv")


register(
    id="forest-fire-v1",
    entry_point="gym_cellular_automata.envs.forest_fire_v1:ForestFireEnv",
)


__all__ = ["Operator", "RESGISTERED_CA_ENVS"]
