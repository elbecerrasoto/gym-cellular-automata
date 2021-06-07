from gym.envs.registration import register

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v1",
)

ffdir = "gym_cellular_automata.forest_fire"

register(
    id=REGISTERED_CA_ENVS[0],
    entry_point=ffdir + ".helicopter:ForestFireEnvHelicopterV0",
    max_episode_steps=999,
)


register(
    id=REGISTERED_CA_ENVS[1],
    entry_point=ffdir + ".bulldozer:ForestFireEnvBulldozerV1",
    max_episode_steps=4096,
    reward_threshold=0.0,
)

__all__ = ["CAEnv", "Operator", "GridSpace", "REGISTERED_CA_ENVS"]
