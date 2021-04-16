from gym.envs.registration import register

from gym_cellular_automata.operator import Operator

RESGISTERED_CA_ENVS = ("ForestFireHelicopter-v0", "ForestFireBulldozer-v0")

register(
    id=RESGISTERED_CA_ENVS[0],
    entry_point="gym_cellular_automata.envs.forest_fire.helicopter_v0:ForestFireEnvHelicopterV0",
)


register(
    id=RESGISTERED_CA_ENVS[1],
    entry_point="gym_cellular_automata.envs.forest_fire.bulldozer_v0:ForestFireEnvBulldozerV0",
)


__all__ = ["Operator", "RESGISTERED_CA_ENVS"]
