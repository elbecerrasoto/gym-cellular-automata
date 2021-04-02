import pytest
import numpy as np

from gym_cellular_automata.envs.forest_fire.v0.operators import ForestFireModifier
from gym_cellular_automata.envs.forest_fire.v0.utils.config import CONFIG

TEST_GRID = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=np.uint8)

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

CELL_STATES = CONFIG["cell_states"]

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

EFFECTS = CONFIG["effects"]

ACTION_UP_LEFT = CONFIG["actions"]["up_left"]
ACTION_UP = CONFIG["actions"]["up"]
ACTION_UP_RIGHT = CONFIG["actions"]["up_right"]

ACTION_LEFT = CONFIG["actions"]["left"]
ACTION_NOT_MOVE = CONFIG["actions"]["not_move"]
ACTION_RIGHT = CONFIG["actions"]["right"]

ACTION_DOWN_LEFT = CONFIG["actions"]["down_left"]
ACTION_DOWN = CONFIG["actions"]["down"]
ACTION_DOWN_RIGHT = CONFIG["actions"]["down_right"]


def test_forest_fire_helicopter_movement():
    grid = TEST_GRID

    pos = np.array([1, 1])

    modifier = ForestFireModifier(EFFECTS)

    # Up
    row, col = modifier._move(grid, ACTION_UP_LEFT, pos)
    assert row == 0 and col == 0
    row, col = modifier._move(grid, ACTION_UP, pos)
    assert row == 0 and col == 1
    row, col = modifier._move(grid, ACTION_UP_RIGHT, pos)
    assert row == 0 and col == 2

    # Middle
    row, col = modifier._move(grid, ACTION_LEFT, pos)
    assert row == 1 and col == 0
    row, col = modifier._move(grid, ACTION_NOT_MOVE, pos)
    assert row == 1 and col == 1
    row, col = modifier._move(grid, ACTION_RIGHT, pos)
    assert row == 1 and col == 2

    # Down
    row, col = modifier._move(grid, ACTION_DOWN_LEFT, pos)
    assert row == 2 and col == 0
    row, col = modifier._move(grid, ACTION_DOWN, pos)
    assert row == 2 and col == 1
    row, col = modifier._move(grid, ACTION_DOWN_RIGHT, pos)
    assert row == 2 and col == 2


def test_ForestFireModifier_helicopter_movement_boundaries():

    grid = TEST_GRID

    modifier = ForestFireModifier(EFFECTS)

    corner_up_left = np.array([0, 0])
    corner_up_right = np.array([0, 2])
    corner_down_left = np.array([2, 0])
    corner_down_right = np.array([2, 2])

    # Up Corners
    row, col = modifier._move(grid, ACTION_UP_LEFT, corner_up_left)
    assert row == 0 and col == 0
    row, col = modifier._move(grid, ACTION_UP_RIGHT, corner_up_right)
    assert row == 0 and col == 2

    # Down Corners
    row, col = modifier._move(grid, ACTION_DOWN_LEFT, corner_down_left)
    assert row == 2 and col == 0
    row, col = modifier._move(grid, ACTION_DOWN_RIGHT, corner_down_right)
    assert row == 2 and col == 2


def test_ForestFireModifier_helicopter_illegal_actions():
    modifier = ForestFireModifier(EFFECTS)

    with pytest.raises(ValueError):
        pos = np.array([1, 1])
        modifier(TEST_GRID, 0, pos)
        modifier(TEST_GRID, -1, pos)
        modifier(TEST_GRID, 42, pos)
        modifier(TEST_GRID, "foo", pos)


def test_ForestFireModifier_helicopter_fire_extinguish():
    grid = TEST_GRID
    pos = [1, 1]
    modifier = ForestFireModifier(EFFECTS)

    new_grid, (row, col) = modifier(grid, ACTION_UP, pos)
    assert new_grid[row, col] == EMPTY

    assert new_grid is grid, "Same Object"

    new_grid, (row, col) = modifier(grid, ACTION_DOWN, pos)
    assert new_grid[row, col] == EMPTY

    assert new_grid is grid, "Same Object"
