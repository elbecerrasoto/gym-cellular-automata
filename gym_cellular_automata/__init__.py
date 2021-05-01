from gym.envs.registration import register

from gym_cellular_automata.operator import Operator

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v0",
    "ForestFireBulldozerEasy-v0",
)

register(
    id=REGISTERED_CA_ENVS[0],
    entry_point="gym_cellular_automata.envs.forest_fire.helicopter_v0:ForestFireEnvHelicopterV0",
)


register(
    id=REGISTERED_CA_ENVS[1],
    entry_point="gym_cellular_automata.envs.forest_fire.bulldozer_v0:ForestFireEnvBulldozerV0",
)


__all__ = ["Operator", "REGISTERED_CA_ENVS"]
