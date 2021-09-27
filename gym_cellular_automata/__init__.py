import warnings
from pathlib import Path

from gym.envs.registration import register
from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
warnings.filterwarnings("ignore", message="Glyph 108 missing from current font.")
warnings.filterwarnings("ignore", message="Glyph 112 missing from current font.")

# Global path on current machine
PROJECT_PATH = Path(__file__).parents[1]

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter5x5-v1",
    "ForestFireBulldozer-v1",
)

try:
    ffdir = "gym_cellular_automata.forest_fire"
    register(
        id=REGISTERED_CA_ENVS[0],
        entry_point=ffdir + ".helicopter:ForestFireEnvHelicopterV0",
        kwargs={"nrows": 5, "ncols": 5},
    )

    register(
        id=REGISTERED_CA_ENVS[1],
        entry_point=ffdir + ".bulldozer:ForestFireEnvBulldozerV1",
    )
except GymError:  # Avoid annoying Re-register error when working interactively.
    pass


__all__ = ["CAEnv", "Operator", "GridSpace", "REGISTERED_CA_ENVS"]
