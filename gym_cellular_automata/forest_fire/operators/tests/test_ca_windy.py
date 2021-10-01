import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.operators.ca_windy import WindyForestFire
from gym_cellular_automata.forest_fire.utils.neighbors import neighborhood_at

# Number of random grids to test
TESTS = 16

# Steps to check CA rules per test grid
STEPS = 4

# Cells checked per step
CHECKS_PER_STEP = 4

# Cell values
EMPTY = 0
BURNED = 1
TREE = 3
FIRE = 25

# Test Grid Size
ROW = 4
COL = 4


@pytest.fixture
def ca():
    return WindyForestFire(EMPTY, BURNED, TREE, FIRE)


# Deterministic Wind
@pytest.fixture
def wind(ca):
    return ca.context_space.high


@pytest.fixture
def grid_space():
    return GridSpace(
        values=[EMPTY, BURNED, TREE, FIRE],
        shape=(ROW, COL),
    )


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


def test_CAwindy_is_operator(ca):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(ca)


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

    # FIRE -> BURNED
    if old_cell_value == FIRE:
        assert new_cell_value == BURNED, "FIRE Consumption (failed)" + log_error

    # Implicit rules (Staying the same)

    # EMPTY -> EMPTY
    if old_cell_value == EMPTY:
        assert new_cell_value == EMPTY, "EMPTY is EMPTY forever (failed)" + log_error

    # BURNED -> BURNED
    if old_cell_value == BURNED:
        assert new_cell_value == BURNED, "BURNED is BURNED forever (failed)" + log_error
