from pathlib import Path

from gym.envs.registration import register
from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

# Global path on current machine
PROJECT_PATH = Path(__file__).parents[1]


REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v1",
)

try:
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
except GymError:  # Avoid annoying Re-register error when working interactively.
    pass


def get_path(file, pwd, behind=1):
    """
    Absolute Path of file,
    expected to be found 1 directory behind of Current Working Directory

        >>> get_path("my_file.yaml", __file__)
    """
    from pathlib import Path

    dir = Path(pwd).parents[behind]
    return dir / file


__all__ = ["CAEnv", "Operator", "GridSpace", "REGISTERED_CA_ENVS"]
