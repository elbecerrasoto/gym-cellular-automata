import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.operators.move import Move

TEST_REPETITIONS = 16

ACTIONS = 9

UP_LEFT, UP, UP_RIGHT, LEFT, NOT_MOVE, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT = range(
    ACTIONS
)

ROW = 3
COL = 3


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
    return GridSpace(n=1, shape=(ROW, COL))


@pytest.fixture
def action_space():
    return spaces.Discrete(ACTIONS)


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.mark.repeat(TEST_REPETITIONS)
def test_move(move, grid_space, action_space, position_space, directions_sets):

    up_set = directions_sets["up"]
    down_set = directions_sets["down"]
    left_set = directions_sets["left"]
    right_set = directions_sets["right"]

    grid = grid_space.sample()
    n_row, n_col = grid.shape

    action = action_space.sample()

    context = position_space.sample()
    row, col = context

    # fmt: off
    if (action in up_set)    and (row > 0):
        row -= 1

    if (action in down_set)  and (row < (n_row-1)):
        row += 1

    if (action in left_set)  and (col > 0):
        col -= 1

    if (action in right_set) and (col < (n_col-1)):
        col += 1
    # fmt: on

    expected_position = np.array([row, col])

    grid, observed_position = move(grid, action, context)

    assert np.all(observed_position == expected_position)


def test_move_warnings(
    move,
    grid_space,
    position_space,
):
    """
    Move operator
    White-box Warning Tests
    """

    # Magic Variables
    unhashable = np.array(0)
    out_of_action_space = "Arbitrary Object"

    grid = grid_space.sample()

    context = position_space.sample()

    with pytest.warns(UserWarning):
        # Unhashable Warning
        action = unhashable
        move(grid, action, context)

        # Out of action space Warning
        action = out_of_action_space
        move(grid, action, context)
