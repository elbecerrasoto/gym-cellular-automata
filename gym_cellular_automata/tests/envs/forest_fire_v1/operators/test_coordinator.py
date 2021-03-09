import pytest

from gym import spaces
import numpy as np

from gym_cellular_automata.envs.forest_fire_v1.operators import (
    WindyForestFire,
    Bulldozer,
    Coordinator,
)

from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

MAX_FREEZE = 64
WIND_TYPE = np.float64
CELL_TYPE = CONFIG["cell_type"]

# Test Grid Size
ROW = 8
COL = 8

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

# Random grid init probabilities
P_EMPTY = 0.20
P_TREE = 0.70
P_FIRE = 0.10


CA_PARAMS_SPACE = spaces.Box(0.0, 1.0, shape=(3, 3))
BD_PARAMS_SPACE = spaces.MultiDiscrete([ROW, COL])

GRID_SPACE = Grid(
    values=[EMPTY, TREE, FIRE], shape=(ROW, COL), probs=[P_EMPTY, P_TREE, P_FIRE]
)

ACTION_SPACE = spaces.MultiDiscrete([9, 2])


@pytest.fixture
def ca():
    return WindyForestFire()


@pytest.fixture
def bulldozer():
    return Bulldozer(CONFIG["effects"])


@pytest.fixture
def coordinator(ca, bulldozer):
    return Coordinator(ca, bulldozer, max_freeze=MAX_FREEZE)


def test_API(coordinator):
    from gym_cellular_automata.tests import test_Operator_API_specifications

    test_Operator_API_specifications(coordinator)


# Test integration
# RE check on Env


# What things do I want to ask from this dude?

# Basically two things

# 0. This is testing the integration Windy-Bulldozer
# I have not done that.

# 1. On freeze != 0
# On the grid at most changes on 1 cell change (and I know which one should be)

# 2. On freeze == 0
# White box test?!
# Manually doing the ca -> bulldozer

# The grid changes into
# Dying -> ~Rules
# Test condition:
# IF D THEN ~R
# ELSE R
# Dying xor Rules

# Test integration
