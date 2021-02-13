from gym.envs.registration import register
from gym_cellular_automata.operator import Operator

RESGISTERED_CA_ENVS = ("forest-fire-v0",)

register(
    id="forest-fire-v0", entry_point="gym_cellular_automata.envs:ForestFireEnv",
)

__all__ = ["Operator", "RESGISTERED_CA_ENVS"]
