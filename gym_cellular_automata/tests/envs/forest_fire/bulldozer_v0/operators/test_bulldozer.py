from operator import mul
from functools import reduce

import pytest
import numpy as np

from gym_cellular_automata.envs.forest_fire.v1.operators.bulldozer import Bulldozer
from gym_cellular_automata.envs.forest_fire.v1.utils.config import CONFIG


TEST_ROW = 8
TEST_COL = 8

# fmt: off
EMPTY = CONFIG["cell_symbols"]["empty"]
TREE  = CONFIG["cell_symbols"]["tree"]

NOT_MOVE   = CONFIG["actions"]["movement"]["not_move"]
DOWN_RIGHT = CONFIG["actions"]["movement"]["down_right"]

NONE  = CONFIG["actions"]["shooting"]["none"]
SHOOT = CONFIG["actions"]["shooting"]["shoot"]
# ftm:on

@pytest.fixture
def bulldozer():
    return Bulldozer()


@pytest.fixture
def all_trees_grid():
    shape = TEST_ROW, TEST_COL
    repetitions = reduce(mul, shape)

    tree = np.array(TREE)

    return np.repeat(tree, repetitions).reshape(shape)


@pytest.fixture
def initial_pos():
    return np.array([0, 0])


def test_bulldozing_trees(bulldozer, all_trees_grid, initial_pos):

    # NOT SHOOT and NOT MOVE
    action = np.array([NOT_MOVE, NONE])

    new_grid, new_pos = bulldozer(all_trees_grid, action, initial_pos)

    row, col = new_pos

    assert new_grid[row, col] == TREE
    assert np.all(initial_pos == new_pos)

    # SHOOT while moving DOWN and RIGHT

    action = np.array([DOWN_RIGHT, SHOOT])

    new_grid, new_pos = bulldozer(new_grid, action, new_pos)

    row, col = new_pos

    assert new_grid[row, col] == EMPTY

    manual_move = initial_pos + [1, 1]

    assert np.all(new_pos == manual_move)
