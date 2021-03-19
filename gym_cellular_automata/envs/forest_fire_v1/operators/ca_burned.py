from collections import namedtuple

import numpy as np
from scipy.signal import convolve2d
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

# fmt: off
EMPTY  = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE   = CONFIG["cell_symbols"]["tree"]
FIRE   = CONFIG["cell_symbols"]["fire"]
# fmt: on

# Signal Weights
BASE = 2

I_EXP = 11
P_EXP = 3

IDENTITY = BASE ** I_EXP
PROPAGATION = BASE ** P_EXP

# Kernel Size
ROW_K = 3
COL_K = 3


# ------------ Correctness


def assert_correctness():

    assert ROW_K == 3, "Only Moore's neighborhood"
    assert COL_K == 3, "Only Moore's neighborhood"

    # Neighbors without including the current cell
    n = 8

    # Weights
    i = IDENTITY
    p = PROPAGATION

    # Values
    E = EMPTY
    B = BURNED
    T = TREE
    F = FIRE

    # Ordering of cell values
    assert E < B
    assert B < T
    assert T < F

    # Ordering of Weights
    assert p < i

    # Test the boundaries of the 5 intervals
    # Interval Names:
    # Unborn, Dead, Keep, Propagate, Consume

    worst = n * p * F

    assert i * E + worst < i * B, "Unborn / Dead"
    assert i * B + worst < i * T, "Dead / Keep"

    # Key Assert, a TREE cell is subject to two different rules
    assert i * T + n * p * T < i * T + p * F, "Keep / Propagate"

    assert i * T + worst < i * F, "Propagate / Consume"


assert_correctness()


# ------------ Breaks


def get_breaks():
    """
    4 breaks needed for 5 rules.
    """

    # "Unborn / Dead"
    dead_break = IDENTITY * BURNED

    # "Dead / Keep"
    keep_break = IDENTITY * TREE

    # "Keep / Propagate"
    propagate_break = IDENTITY * TREE + PROPAGATION * FIRE

    # "Propagate / Consume"
    consume_break = IDENTITY * FIRE

    Breaks = namedtuple("Breaks", ["dead", "keep", "propagate", "consume"])

    return Breaks(dead_break, keep_break, propagate_break, consume_break)


BREAKS = get_breaks()


# ------------ Forest Fire Cellular Automaton


class WindyForestFireB(Operator):
    is_composition = False

    def __init__(self, grid_space=None, action_space=None, context_space=None):

        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(3, 3))

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        # Context contains Wind parameters
        # Sample which FIREs fail to propagate this update
        fail_to_propagate = get_failed_propagations_mask(context)

        kernel = get_kernel(fail_to_propagate)

        grid_signal = convolve(grid, kernel)

        new_grid = translate_analogic_to_discrete(grid_signal, BREAKS)

        return new_grid, context


# ------------ Utils


def get_failed_propagations_mask(wind):
    """
    Here goes the only sampling of the step.
    """
    uniform_space = spaces.Box(low=0.0, high=1.0, shape=(ROW_K, COL_K))
    uniform_roll = uniform_space.sample()

    failed_propagations = np.repeat(False, ROW_K * COL_K).reshape(ROW_K, COL_K)
    failed_propagations = wind <= uniform_roll

    return failed_propagations


def get_kernel(failed_propagations):
    kernel = np.repeat(PROPAGATION, ROW_K * COL_K).reshape(ROW_K, COL_K)

    kernel[failed_propagations] = EMPTY
    kernel[1, 1] = IDENTITY

    return kernel


def convolve(grid, kernel):
    return convolve2d(grid, kernel, mode="same", boundary="fill", fillvalue=EMPTY)


def translate_analogic_to_discrete(grid, breaks):
    row, col = grid.shape

    # Init on empty by default
    empty = np.array(EMPTY)
    new_grid = np.repeat(empty, row * col).reshape(row, col)

    # 5 Rules to carry out:

    # 1. Unborn
    # EMPTY -> EMPTY

    # Implicitly defined by default values
    unborn_mask = grid < breaks.dead
    new_grid[unborn_mask] = EMPTY

    # 2. Dead
    # BURNED -> BURNED
    dead_mask = np.logical_and(grid >= breaks.dead, grid < breaks.keep)
    new_grid[dead_mask] = BURNED

    # 3. Keep
    # TREE -> TREE
    keep_mask = np.logical_and(grid >= breaks.keep, grid < breaks.propagate)
    new_grid[keep_mask] = TREE

    # 4. Propagate
    # TREE -> FIRE
    propagate_mask = np.logical_and(grid >= breaks.propagate, grid < breaks.consume)
    new_grid[propagate_mask] = FIRE

    # 4. Consume
    # FIRE -> BURNED
    propagate_mask = grid >= breaks.consume
    new_grid[propagate_mask] = BURNED

    return new_grid
