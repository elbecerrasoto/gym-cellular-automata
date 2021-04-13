import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata.envs.forest_fire.operators.modify import Modify
from gym_cellular_automata.grid_space import Grid

TEST_REPETITIONS = 16


# Arbitrary cell values
# Set to those numbers to be consistent with the following implementation
# ... refer to file
# For simplicity they could be changed to something like
# EMPTY = 0; TREE = 1; FIRE = 2
EMPTY = 0
TREE = 3
FIRE = 25

# Test Grid size
ROW = 8
COL = 8


@pytest.fixture
def effects():
    return {EMPTY: TREE, TREE: FIRE, FIRE: EMPTY}


@pytest.fixture
def modify(effects):
    return Modify(effects)


@pytest.fixture
def grid_space():
    return Grid(values=[EMPTY, TREE, FIRE], shape=(ROW, COL))


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.mark.repeat(TEST_REPETITIONS)
def test_modify_cell_at_position(modify, effects, grid_space, position_space):

    for shoot_action in {True, False}:

        random_grid = grid_space.sample()
        action = None, shoot_action
        random_position = position_space.sample()

        row, col = random_position
        target_cell = random_grid[row, col]

        expected_cell = effects[target_cell] if shoot_action else target_cell

        grid, position = modify(random_grid, action, random_position)

        observed_cell = grid[row, col]

        assert observed_cell == expected_cell
        assert np.all(random_position == position)
