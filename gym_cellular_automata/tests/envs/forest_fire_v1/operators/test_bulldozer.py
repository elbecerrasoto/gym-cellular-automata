from operator import mul
from functools import reduce

import pytest
import numpy as np

from gym_cellular_automata.envs.forest_fire_v1.operators import bulldozer as bd
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

from gym_cellular_automata.tests import test_Operator_API_specifications


TEST_ROW = 8
TEST_COL = 8

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]


@pytest.fixture
def bulldozer():
    return bd.Bulldozer(CONFIG["effects"])


@pytest.fixture
def all_trees_grid():
    shape = TEST_ROW, TEST_COL
    repetitions = reduce(mul, shape)

    fire = np.array(TREE, dtype=CONFIG["cell_type"])

    return np.repeat(fire, repetitions).reshape(shape)


@pytest.fixture
def initial_pos():
    return np.array([0, 0])


def test_API(bulldozer):
    test_Operator_API_specifications(bulldozer)


def test_bulldozing_trees(bulldozer, all_trees_grid, initial_pos):

    # NOT SHOOT and NOT MOVE
    action = np.array([bd.NOT_MOVE, bd.NONE])

    new_grid, new_pos = bulldozer(all_trees_grid, action, initial_pos)

    row, col = new_pos

    assert new_grid[row, col] == TREE
    assert np.all(initial_pos == new_pos)

    # SHOOT while moving DOWN and RIGHT

    action = np.array([bd.DOWN_RIGHT, bd.SHOOT])

    new_grid, new_pos = bulldozer(new_grid, action, new_pos)

    row, col = new_pos

    assert new_grid[row, col] == EMPTY

    manual_move = initial_pos + [1, 1]

    assert np.all(new_pos == manual_move)
