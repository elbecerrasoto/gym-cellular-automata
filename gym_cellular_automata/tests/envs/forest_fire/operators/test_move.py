import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata.envs.forest_fire.operators.move import Move
from gym_cellular_automata.grid_space import Grid

TEST_REPETITIONS = 16

MOVE_ACTIONS = 9
SHOOT_ACTIONS = 2


UP_LEFT, UP, UP_RIGHT, LEFT, NOT_MOVE, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT = range(
    MOVE_ACTIONS
)

ROW = 3
COL = 3


@pytest.fixture
def directions_sets():
    return {
        "up_set": {UP_LEFT, UP, UP_RIGHT},
        "down_set": {DOWN_LEFT, DOWN, DOWN_RIGHT},
        "left_set": {UP_LEFT, LEFT, DOWN_LEFT},
        "right_set": {UP_RIGHT, RIGHT, DOWN_RIGHT},
        "not_move_set": {NOT_MOVE},
    }


@pytest.fixture
def move(directions_sets):
    return Move(**directions_sets)


@pytest.fixture
def grid_space():
    return Grid(n=1, shape=(ROW, COL))


@pytest.fixture
def action_space():
    return spaces.MultiDiscrete([MOVE_ACTIONS, SHOOT_ACTIONS])


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.mark.repeat(TEST_REPETITIONS)
def test_move(move, grid_space, action_space, position_space, directions_sets):

    up_set = directions_sets["up_set"]
    down_set = directions_sets["down_set"]
    left_set = directions_sets["left_set"]
    right_set = directions_sets["right_set"]

    grid = grid_space.sample()
    n_row, n_col = grid.shape

    action = action_space.sample()
    move_action, shoot_action = action

    context = position_space.sample()
    row, col = context

    # fmt: off
    if (move_action in up_set)    and (row > 0):
        row -= 1

    if (move_action in down_set)  and (row < (n_row-1)):
        row += 1

    if (move_action in left_set)  and (col > 0):
        col -= 1

    if (move_action in right_set) and (col < (n_col-1)):
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
    shoot_action = None

    grid = grid_space.sample()

    context = position_space.sample()

    with pytest.warns(UserWarning):
        # Unhashable Warning
        action = unhashable, shoot_action
        move(grid, action, context)

        # Out of action space Warning
        action = out_of_action_space, shoot_action
        move(grid, action, context)
