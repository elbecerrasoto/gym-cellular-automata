import pytest
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_BOX, TYPE_INT
from gym_cellular_automata.forest_fire.operators.ca_DrosselSchwabl import ForestFire
from gym_cellular_automata.forest_fire.utils.neighbors import neighborhood_at
from gym_cellular_automata.grid_space import GridSpace

# Number of random grids to test
TESTS = 16

# Steps to check CA rules per test grid
STEPS = 4

# Cells checked per step
CHECKS_PER_STEP = 4

# Cell values
EMPTY, TREE, FIRE = range(3)

# Grid size
ROW, COL = 4, 4


@pytest.fixture
def grid_space():
    return GridSpace(
        values=[EMPTY, TREE, FIRE],
        shape=(ROW, COL),
    )


@pytest.fixture
def ca_params_space():
    return spaces.Box(0.0, 1.0, shape=(2,), dtype=TYPE_BOX)


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL], dtype=TYPE_INT)


@pytest.fixture
def ca(grid_space, ca_params_space):
    dummy_space = ca_params_space
    return ForestFire(
        EMPTY,
        TREE,
        FIRE,
        grid_space=grid_space,
        action_space=dummy_space,
        context_space=ca_params_space,
    )


def test_drosselSchwabl_is_operator(ca):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(ca, strict=False)


@pytest.mark.repeat(TESTS)
def test_forest_fire_update(ca, grid_space, ca_params_space, position_space):
    grid = grid_space.sample()
    ca_params = ca_params_space.sample()

    for step in range(STEPS):
        new_grid, __ = ca(grid, None, ca_params)

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

    if old_cell_value == TREE:
        # TREE -> FIRE (propagate)
        if FIRE in neighborhood:
            assert new_cell_value == FIRE, "TREE -> FIRE (failed)" + log_error

        # TREE -> TREE | FIRE (keep or strike)
        else:
            assert new_cell_value != EMPTY, "TREE -> TREE | FIRE (failed)" + log_error

    # FIRE -> EMPTY
    if old_cell_value == FIRE:
        assert new_cell_value == EMPTY, "FIRE -> EMPTY (failed)" + log_error

    # EMPTY -> EMPTY | TREE
    if old_cell_value == EMPTY:
        assert new_cell_value != FIRE, "EMPTY -> EMPTY | TREE (failed)" + log_error
