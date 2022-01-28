"""
Gym Cellular Automata Enviroments
==================================

Gym Cellular Automata is a collection of
Reinforcement Learning Environments (RLEs)
that follow the OpenAI Gym API.

The available RLEs are based on Cellular Automata (CAs).
On them an Agent interacts with a CA,
by changing its cell states,
in a attempt to drive the emergent properties
of its grid to a desired configuration.

See https://github.com/elbecerrasoto/gym-cellular-automata for documentation.
"""
from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator
from gym_cellular_automata.registration import GYM_MAKE
from gym_cellular_automata.registration import GYM_MAKE as envs
from gym_cellular_automata.registration import (
    REGISTERED_CA_ENVS,
    prototypes,
    register_caenvs,
)
from gym_cellular_automata.version import VERSION as __version__

# Do NOT import anything from here
# otherwise a circular import will be triggered
# These are exports for external code


# Avoids annoying error when working interactively
try:
    register_caenvs()
except GymError:
    pass


__all__ = ["REGISTERED_CA_ENVS", "GYM_MAKE", "CAEnv", "GridSpace", "Operator", "envs"]
