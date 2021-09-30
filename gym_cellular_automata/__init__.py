from pathlib import Path
from warnings import filterwarnings

from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator
from gym_cellular_automata.registration import (
    GYM_MAKE,
    REGISTERED_CA_ENVS,
    register_caenvs,
)

# Global path on current machine
PROJECT_PATH = Path(__file__).parents[1]

try:
    register_caenvs()
except GymError:
    pass


__all__ = ["GYM_MAKE", "REGISTERED_CA_ENVS", "CAEnv", "GridSpace", "Operator"]

# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
filterwarnings("ignore", message="Glyph 108 missing from current font.")
filterwarnings("ignore", message="Glyph 112 missing from current font.")
