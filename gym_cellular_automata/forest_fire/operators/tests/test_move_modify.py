import numpy as np
import pytest
from gymnasium import spaces

from gym_cellular_automata.forest_fire.operators import Modify, Move
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.tests import assert_operator

TEST_REPETITIONS = 16

ACTIONS = 9

UP_LEFT, UP, UP_RIGHT, LEFT, NOT_MOVE, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT = range(
    ACTIONS
)

ROW = 3
COL = 3

CELL_STATES = 3


@pytest.fixture
def directions_sets():
    return {
        "up": {UP_LEFT, UP, UP_RIGHT},
        "down": {DOWN_LEFT, DOWN, DOWN_RIGHT},
        "left": {UP_LEFT, LEFT, DOWN_LEFT},
        "right": {UP_RIGHT, RIGHT, DOWN_RIGHT},
        "not_move": {NOT_MOVE},
    }


@pytest.fixture
def move(directions_sets):
    return Move(directions_sets)


@pytest.fixture
def grid_space():
    return GridSpace(n=3, shape=(ROW, COL))


@pytest.fixture
def action_space():
    return spaces.Discrete(ACTIONS)


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


def test_move_is_operator(move):
    assert_operator(move)


@pytest.mark.repeat(TEST_REPETITIONS)
def test_move(move, grid_space, action_space, position_space, directions_sets):
    up_set = directions_sets["up"]
    down_set = directions_sets["down"]
    left_set = directions_sets["left"]
    right_set = directions_sets["right"]

    grid = grid_space.sample()
    nrows, ncols = grid.shape

    action = action_space.sample()

    context = position_space.sample()
    row, col = context

    expected_position = np.array(
        new_position(row, col, action, grid, up_set, down_set, left_set, right_set)
    )

    grid, observed_position = move(grid, action, context)

    assert np.all(observed_position == expected_position)


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


def test_modify_is_operator(modify):
    assert_operator(modify, strict=False)


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


### Orthogonal new position test
### Orthogonal: different from the method used on library implementation


def new_position(row, col, action, grid, up_set, down_set, left_set, right_set):
    def are_my_neighbors_a_boundary(grid, pos):
        """
        Check if the neighbors of target position are a boundary.
        Return a tuple of Bools informing which neighbor is a boundary.
        It checks the up, down, left, and right neighbors.
        """
        from collections import namedtuple

        row, col = pos
        n_row, n_col = grid.shape

        up_offset, down_offset = row + np.array([-1, 1])
        left_offset, right_offset = col + np.array([-1, 1])

        up = bool(up_offset < 0)
        down = bool(down_offset > n_row - 1)
        left = bool(left_offset < 0)
        right = bool(right_offset > n_col - 1)

        Boundaries = namedtuple("Boundaries", ["up", "down", "left", "right"])

        return Boundaries(up, down, left, right)

    is_boundary = are_my_neighbors_a_boundary(grid, (row, col))

    new_row = (
        row - 1
        if not is_boundary.up and int(action) in up_set
        else row + 1 if not is_boundary.down and int(action) in down_set else row
    )

    new_col = (
        col - 1
        if not is_boundary.left and int(action) in left_set
        else col + 1 if not is_boundary.right and int(action) in right_set else col
    )

    return new_row, new_col
