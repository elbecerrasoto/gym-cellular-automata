"""
from operator import mul
from functools import reduce

import pytest

from gym import spaces
import numpy as np

from gym_cellular_automata.envs.forest_fire_v1.operators import (
    WindyForestFire,
    Bulldozer,
    Coordinator,
)


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


# fmt: off
CA_PARAMS_SPACE = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=WIND_TYPE)
BD_PARAMS_SPACE = spaces.MultiDiscrete([ROW, COL])
GRID_SPACE      = spaces.Box(0, 10, shape=(ROW, COL), dtype=CELL_TYPE)

ACTION_SPACE    = spaces.MultiDiscrete([9, 2])
# fmt: on


# Fixture must be constant data, not reusable functions

pytest.fixture()




# arg scope
# the scope for which this fixture is shared, one of "function" (default), "class", "module", "package" or "session".

# "package" is considered experimental at this time.


def random_grid():
    shape = (ROW, COL)
    size = reduce(mul, shape)
    probs = [P_EMPTY, P_TREE, P_FIRE]
    cell_values = np.array([EMPTY, TREE, FIRE], dtype=CELL_TYPE)

    return np.random.choice(cell_values, size, probs).reshape(shape)


@pytest.fixture
def ca():
    return WindyForestFire()


@pytest.fixture
def bulldozer():
    return Bulldozer(CONFIG["effects"])


@pytest.fixture
def coordinator(ca, bulldozer):
    return Coordinator(ca, bulldozer, max_freeze=MAX_FREEZE)


@pytest.fixture
def sample_coordinator_input(coordinator):

    grid = random_grid()
    action = coordinator.action_space.sample()
    context = coordinator.context_space.sample()

    return grid, action, context


def context_with_custom_freeze(custom=1):
    ca_params = CA_PARAMS_SPACE.sample()
    pos = POS_SPACE.sample()
    freeze = custom

    return ca_params, pos, freeze






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
"""


