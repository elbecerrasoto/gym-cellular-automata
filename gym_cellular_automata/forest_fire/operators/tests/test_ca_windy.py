import pytest
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_INT
from gym_cellular_automata.forest_fire.operators.ca_windy import WindyForestFire
from gym_cellular_automata.forest_fire.utils.neighbors import neighborhood_at
from gym_cellular_automata.grid_space import GridSpace

# Number of random grids to test
TESTS = 16

# Steps to check CA rules per test grid
STEPS = 4

# Cells checked per step
CHECKS_PER_STEP = 4

# Cell values
EMPTY = 0
TREE = 3
FIRE = 25

# Test Grid Size
ROW = 4
COL = 4


@pytest.fixture
def ca():
    return WindyForestFire(EMPTY, TREE, FIRE)


# Deterministic Wind
@pytest.fixture
def wind(ca):
    return ca.context_space.high


@pytest.fixture
def grid_space():
    return GridSpace(values=[EMPTY, TREE, FIRE], shape=(ROW, COL), dtype=TYPE_INT)


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL], dtype=TYPE_INT)


def test_CAwindy_is_operator(ca):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(ca, strict=False)


@pytest.mark.repeat(TESTS)
def test_windy_forest_fire_update(ca, grid_space, wind, position_space):
    grid = grid_space.sample()

    for step in range(STEPS):
        new_grid, __ = ca(grid, None, wind)

        for check in range(CHECKS_PER_STEP):
            row, col = position_space.sample()

            assert_forest_fire_update_at_positionrows_col(grid, new_grid, row, col)

        grid = new_grid


def assert_forest_fire_update_at_positionrows_col(grid, new_grid, row, col):
    log_error = (
        f"\n row: {row}"
        + f"\n col: {col}"
        + f"\n\n grid: {grid}"
        + f"\n\n new_grid: {new_grid}"
    )

    old_cell_value = grid[row, col]
    new_cell_value = new_grid[row, col]

    neighborhood = neighborhood_at(grid, (row, col), invariant=EMPTY)

    # Explicit rules

    if old_cell_value == TREE:
        # TREE -> FIRE (propagate)
        if FIRE in neighborhood:
            assert new_cell_value == FIRE, "FIRE Propagation (failed)" + log_error

        # TREE -> TREE (keep)
        else:
            assert new_cell_value == TREE, "Keep some TREE (failed)" + log_error

    # FIRE -> EMPTY
    if old_cell_value == FIRE:
        assert new_cell_value == EMPTY, "FIRE Consumption (failed)" + log_error

    # Implicit rules (Staying the same)

    # EMPTY -> EMPTY
    if old_cell_value == EMPTY:
        assert new_cell_value == EMPTY, "EMPTY is EMPTY forever (failed)" + log_error
