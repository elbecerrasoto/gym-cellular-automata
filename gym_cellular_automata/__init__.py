from pathlib import Path
from warnings import filterwarnings

import numpy as np
from gym.error import Error as GymError
from gym.spaces import Box

from gym_cellular_automata.version import VERSION as __version__
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


# Delegation of explicit typing as much as possible
# For floats using the spaces Box default
TYPE_BOX = np.float32

# Avoids annoying error when working interactively
try:
    register_caenvs()
except GymError:
    pass


__all__ = [
    "GYM_MAKE",
    "REGISTERED_CA_ENVS",
    "CAEnv",
    "GridSpace",
    "Operator",
    "TYPE_BOX",
]

# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
filterwarnings("ignore", message="Glyph 108 missing from current font.")
filterwarnings("ignore", message="Glyph 112 missing from current font.")
