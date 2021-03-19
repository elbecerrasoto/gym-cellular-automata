from math import isclose
import pytest

from gym_cellular_automata.envs.forest_fire_v1.operators import WindyForestFireB
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire.utils.neighbors import neighborhood_at

from gym import spaces


# Number of random grids to test
TESTS = 32

# Steps to check CA rules per test grid
STEPS = 8

# Cells checked per step
CHECKS_PER_STEP = 8

# Test Grid Size
ROW = 8
COL = 8

# fmt: on
# Random grid init probabilities
P_EMPTY = 0.30
P_BURNED = 0.00
P_TREE = 0.60
P_FIRE = 0.10

assert isclose(
    sum((P_EMPTY, P_BURNED, P_TREE, P_FIRE)), 1.0
), "Grid Init. Probs. must sum to 1.0"

EMPTY = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]
# fmt: on


@pytest.fixture
def ca():
    return WindyForestFireB()


@pytest.fixture
def wind_deterministic(ca):
    return ca.context_space.high


@pytest.fixture
def grid_space():
    return Grid(
        values=[EMPTY, BURNED, TREE, FIRE],
        probs=[P_EMPTY, P_BURNED, P_TREE, P_FIRE],
        shape=(ROW, COL),
    )


@pytest.fixture
def pos_space():
    return spaces.MultiDiscrete([ROW, COL])


def assert_forest_fire_update_at_position_row_col(grid, new_grid, row, col):
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

    if old_cell_value == TREE and FIRE in neighborhood:

        # TREE -> FIRE (conditional)
        assert new_cell_value == FIRE, "FIRE Propagation (failed)" + log_error

    if old_cell_value == FIRE:

        # FIRE -> BURNED
        assert new_cell_value == BURNED, "FIRE Consumption (failed)" + log_error

    # Implicit rules (Staying the same)

    if old_cell_value == EMPTY:

        # EMPTY -> EMPTY
        assert new_cell_value == EMPTY, "EMPTY is EMPTY forever (failed)" + log_error

    if old_cell_value == BURNED:

        # BURNED -> BURNED
        assert new_cell_value == TREE, "BURNED is BURNED forever (failed)" + log_error

    if old_cell_value == TREE:

        # TREE -> TREE (conditional)
        assert new_cell_value == TREE, "Keep some TREE (failed)" + log_error


def visual_inspection(steps: int = 12, sleep: int = 1) -> None:
    import time

    ca = WindyForestFireB()

    grid_space = Grid(
        values=[EMPTY, BURNED, TREE, FIRE],
        probs=[P_EMPTY, P_BURNED, P_TREE, P_FIRE],
        shape=(ROW, COL),
    )

    grid = grid_space.sample()
    wind = ca.context_space.high

    for step in range(steps):
        print(f"Grid at step {step}")
        print(f"{grid}\n")

        grid, wind = ca(grid, None, wind)

        time.sleep(sleep)
