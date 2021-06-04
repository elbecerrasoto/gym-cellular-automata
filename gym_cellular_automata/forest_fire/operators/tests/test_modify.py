import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.operators.modify import Modify

TEST_REPETITIONS = 16

CELL_STATES = 3

# Test Grid size
ROW = 3
COL = 3


@pytest.fixture
def effects():
    return {
        cell_state: range(CELL_STATES)[cell_state - (CELL_STATES - 1)]
        for cell_state in range(CELL_STATES)
    }


@pytest.fixture
def modify(effects):
    return Modify(effects)


@pytest.fixture
def grid_space():
    return GridSpace(n=3, shape=(ROW, COL))


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.mark.repeat(TEST_REPETITIONS)
def test_modify_cell_at_position(modify, effects, grid_space, position_space):

    for action in {True, False}:

        random_grid = grid_space.sample()
        random_position = position_space.sample()

        row, col = random_position
        target_cell = random_grid[row, col]

        expected_cell = effects[target_cell] if action else target_cell

        grid, position = modify(random_grid, action, random_position)

        observed_cell = grid[row, col]

        assert observed_cell == expected_cell
        assert np.all(random_position == position)
