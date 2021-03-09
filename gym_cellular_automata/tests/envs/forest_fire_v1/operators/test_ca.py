from operator import mul
from functools import reduce

import pytest
import numpy as np
from gym import spaces

from gym_cellular_automata.envs.forest_fire_v1.operators import ca

# Number of random grids to test
N_TESTS = 32

# Steps to check CA rules per grid
T_STEPS = 8

# Cells checked per step
CHECKS_PER_STEP = 8

# Test Grid Size
TEST_ROW = 8
TEST_COL = 8

# Random grid init probabilities
P_EMPTY = 0.20
P_TREE = 0.70
P_FIRE = 0.10


# def convolve(grid, kernel):
# Test not shrinking
# Test max values
# Test min values

# def translate_analogic_to_discrete(grid, breaks):
# ??
    
# Extras
# Test operator ABC
# Profile on big grids


@pytest.fixture
def windy_forest_fire():
    return ca.WindyForestFire()


def random_grid():
    shape = (TEST_ROW, TEST_COL)
    size = reduce(mul, shape)
    probs = [P_EMPTY, P_TREE, P_FIRE]
    cell_values = np.array([ca.EMPTY, ca.TREE, ca.FIRE], dtype=ca.CELL_TYPE)

    return np.random.choice(cell_values, size, p=probs).reshape(shape)


@pytest.fixture
def pos_space():
    return spaces.MultiDiscrete([TEST_ROW, TEST_COL])


@pytest.fixture
def uniform_space():
    return spaces.Box(0.0, 1.0, shape=(ca.ROW_K, ca.COL_K), dtype=ca.WIND_TYPE)


@pytest.fixture
def deterministic_winds(uniform_space):
    return uniform_space.high, uniform_space.low


def test_failed_propagations(deterministic_winds):
    certain, impossible = deterministic_winds

    all_false = ca.get_failed_propagations_mask(certain)
    all_true = ca.get_failed_propagations_mask(impossible)

    assert not np.all(all_false)
    assert np.all(all_true)


def test_kernel_generation(deterministic_winds, uniform_space):
    certain, impossible = deterministic_winds
    unif_sampling = uniform_space.sample()
    
    all_propagating_kernel = ca.get_kernel(ca.get_failed_propagations_mask(certain))
    none_propagating_kernel = ca.get_kernel(ca.get_failed_propagations_mask(impossible))
    
    sampled_kernel = ca.get_kernel(ca.get_failed_propagations_mask(unif_sampling))
    
    kernels = all_propagating_kernel, none_propagating_kernel, sampled_kernel
    
    k_row, k_col = kernels[0].shape

    assert k_row == 3
    assert k_col == 3
    
    def check_kernel_weight_at_position(kernel, position, weight):
        row, col = position
        return kernel[row, col] == weight
    
    for kernel in kernels:
        assert check_kernel_weight_at_position(kernel, (1, 1), ca.IDENTITY)
        
    assert check_kernel_weight_at_position(none_propagating_kernel, (0, 0), ca.EMPTY)
    
    assert check_kernel_weight_at_position(all_propagating_kernel, (0, 0), ca.PROPAGATION)

    assert check_kernel_weight_at_position(all_propagating_kernel, (0, 0), ca.EMPTY) ^ \
        check_kernel_weight_at_position(all_propagating_kernel, (0, 0), ca.PROPAGATION)


def assert_forest_fire_update_at_position_row_col(grid, new_grid, row, col):
    log_error = (
        f"\n row: {row}"
        + f"\n col: {col}"
        + f"\n\n grid: {grid}"
        + f"\n\n new_grid: {new_grid}"
    )

    old_cell_value = grid[row, col]
    new_cell_value = new_grid[row, col]

    # The rule of staying on the same cell state is implicitly tested.
    # Probabilistic rules cannot be black-box tested.

    # Explicit rules
    if old_cell_value == ca.FIRE:
        
        # A FIRE turns into an EMPTY.
        assert new_cell_value == ca.EMPTY, "NON Fire Consumption" + log_error

    # Implicit rules
    if old_cell_value == ca.EMPTY:
        
        # An EMPTY never turns into TREE.
        assert new_cell_value != ca.TREE, "We do not sow" + log_error
        
        # An EMPTY never turns into FIRE.
        assert new_cell_value != ca.FIRE, "Empty Combustion" + log_error


    if old_cell_value == ca.TREE:
        
        # A TREE never turns into EMPTY.
        assert new_cell_value != ca.EMPTY, "Dying Trees" + log_error

    if old_cell_value == ca.FIRE:
        
        # A FIRE never turns into a TREE.
        assert new_cell_value != ca.TREE, "Created by Fire" + log_error
        
        # A FIRE never turns into a FIRE.
        assert new_cell_value != ca.FIRE, "Lingering Fire" + log_error


def test_windy_forest_fire_update(windy_forest_fire, pos_space):

    for i_test in range(N_TESTS):
        
        grid = random_grid()
        
        for step in range(T_STEPS):
            
            wind = windy_forest_fire.context_space.sample()
    
            new_grid, context = windy_forest_fire(grid, None, wind)
    
            assert grid is not new_grid, "Operator is returning the same grid object."
    
            for check in range(CHECKS_PER_STEP):
                
                row, col = pos_space.sample()
    
                assert_forest_fire_update_at_position_row_col(grid, new_grid, row, col)
    
            grid = new_grid
