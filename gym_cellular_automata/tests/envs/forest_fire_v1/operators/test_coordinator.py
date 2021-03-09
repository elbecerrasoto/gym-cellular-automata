import pytest
import numpy as np

from gym_cellular_automata.envs.forest_fire_v1.operators import (
    WindyForestFire,
    Bulldozer,
    Coordinator,
)
from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

MAX_FREEZE = 1
CELL_TYPE = CONFIG["cell_type"]

# Test Grid Size
ROW = 8
COL = 8

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


DOWN_RIGHT = CONFIG["actions"]["movement"]["down_right"]
SHOOT = CONFIG["actions"]["shooting"]["shoot"]
NONE = CONFIG["actions"]["shooting"]["none"]


WIND = CONFIG["wind"]


@pytest.fixture
def ca():
    return WindyForestFire()


@pytest.fixture
def bulldozer():
    return Bulldozer()


@pytest.fixture
def coordinator(ca, bulldozer):
    return Coordinator(ca, bulldozer, max_freeze=MAX_FREEZE)


@pytest.fixture
def init_position():
    return np.array([0, 0])


@pytest.fixture
def fixed_tree_grid(init_position):
    grid_space = Grid(
        values=[EMPTY, TREE, FIRE], shape=(ROW, COL), probs=[0.10, 0.87, 0.03]
    )
    grid = grid_space.sample()

    row, col = init_position

    grid[row + 1, col + 1] = TREE

    return grid


@pytest.fixture
def action_shoot_down_right():
    return np.array([DOWN_RIGHT, SHOOT])


@pytest.fixture
def action_none_down_right():
    return np.array([DOWN_RIGHT, NONE])


def test_API(coordinator):
    from gym_cellular_automata.tests import test_Operator_API_specifications

    test_Operator_API_specifications(coordinator)


def test_coordinator_on_freeze_diff_than_zero(
    coordinator, fixed_tree_grid, action_shoot_down_right, init_position
):

    freeze = 1

    grid = fixed_tree_grid
    action = action_shoot_down_right
    context = WIND, init_position, freeze

    print(f"action drs: {action_shoot_down_right}")
    print(f"init pos {init_position}")

    new_grid, new_context = coordinator(grid, action, context)

    ca_params, mod_params, new_freeze = new_context

    assert new_freeze == freeze - 1

    # Redundant tests of Bulldozer, White-boxy tests, could be removed
    # But whatever, more robustness!
    row, col = mod_params

    assert new_grid[row, col] == EMPTY, "Action was shooting, tree should be removed"
    assert (row, col) == tuple(init_position + 1)

    # Restore the destroyed tree
    new_grid[row, col] = TREE

    # At most one cell changes, however we have restored it
    assert np.all(grid == new_grid)


"""
def test_coordinator_on_freeze_equal_to_zero(
    coordinator, fixed_tree_grid, action_none_down_right, init_position
):

    freeze = 0

    grid = fixed_tree_grid
    action = action_none_down_right
    context = WIND, init_position, freeze

    new_grid, new_context = coordinator(grid, action, context)

    ca_params, mod_params, new_freeze = new_context

    assert new_freeze == MAX_FREEZE, "Freeze reset"

    # 1-step CA update
    ca_updated_grid = coordinator.cellular_automaton(grid, None, WIND)

    # As the bulldozer did NOT cut a tree both grids should be equal
    assert np.all(ca_updated_grid == new_grid)
"""
