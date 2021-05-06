from gym.envs.registration import register

from gym_cellular_automata.operator import Operator

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v0",
    "ForestFireBulldozer-v1",
)

ff_dir = "gym_cellular_automata.envs.forest_fire"

register(
    id=REGISTERED_CA_ENVS[0],
    entry_point=ff_dir + ".helicopter_v0:ForestFireEnvHelicopterV0",
)


register(
    id=REGISTERED_CA_ENVS[1],
    entry_point=ff_dir + ".bulldozer_v0:ForestFireEnvBulldozerV0",
)


register(
    id=REGISTERED_CA_ENVS[2],
    entry_point=ff_dir + ".bulldozer_v1:ForestFireEnvBulldozerV1",
)

__all__ = ["Operator", "REGISTERED_CA_ENVS"]
