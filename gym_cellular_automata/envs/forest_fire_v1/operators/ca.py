from collections import namedtuple

import numpy as np
from scipy.signal import convolve2d
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

# fmt: off
EMPTY = CONFIG["cell_symbols"]["empty"]
TREE  = CONFIG["cell_symbols"]["tree"]
FIRE  = CONFIG["cell_symbols"]["fire"]

CELL_TYPE = CONFIG["cell_type"]

UP_LEFT    = CONFIG["wind"]["up_left"]
UP         = CONFIG["wind"]["up"]
UP_RIGHT   = CONFIG["wind"]["up_right"]
LEFT       = CONFIG["wind"]["left"]
SELF       = CONFIG["wind"]["self"]
RIGHT      = CONFIG["wind"]["right"]
DOWN_LEFT  = CONFIG["wind"]["down_left"]
DOWN       = CONFIG["wind"]["down"]
DOWN_RIGHT = CONFIG["wind"]["down_right"]

WIND = [[UP_LEFT,   UP,   UP_RIGHT  ],
        [LEFT,      SELF, RIGHT     ],
        [DOWN_LEFT, DOWN, DOWN_RIGHT]]

WIND_TYPE = np.float64
# fmt: on

WIND = np.array(WIND, dtype=WIND_TYPE)

# Signal Weights
BASE = 2

I_EXP = 10
P_EXP = 3

IDENTITY = BASE ** I_EXP
PROPAGATION = BASE ** P_EXP

# Kernel Size
ROW_K = 3
COL_K = 3


# ------------ Correctness


def assert_correctness():
    assert EMPTY*IDENTITY + 8*FIRE*PROPAGATION < TREE*IDENTITY, "empty / tree"
    assert TREE*IDENTITY < TREE*IDENTITY + 8*TREE*PROPAGATION, "keep / keep"
    assert TREE*IDENTITY + 8*TREE*PROPAGATION < TREE*IDENTITY + FIRE*PROPAGATION, "keep / burn"
    assert TREE*IDENTITY + FIRE*PROPAGATION < TREE*IDENTITY + 8*FIRE*PROPAGATION, "burn / burn"
    assert TREE*IDENTITY + 8*FIRE*PROPAGATION < FIRE*IDENTITY, "burn / consume"
    assert ROW_K == 3, "Only Moore's neighborhood"
    assert COL_K == 3, "Only Moore's neighborhood"


assert_correctness()


# ------------ Breaks


def get_breaks():
    """ 3 breaks needed for 4 conditions.
    empty < keep < burn < consume
    """
    keep_break = TREE * IDENTITY
    burn_break = TREE * IDENTITY + FIRE * PROPAGATION
    consume_break = FIRE * IDENTITY

    Breaks = namedtuple("Breaks", ["keep_break", "burn_break", "consume_break"])

    return Breaks(keep_break, burn_break, consume_break)

BREAKS = get_breaks()


# ------------ 1-step CA update


def UPDATE(grid):
    # Sample Wind
    fail_to_propagate = get_failed_propagations_mask()
    
    kernel = get_kernel(fail_to_propagate)

    grid_signal = convolve(grid, kernel)

    return translate_analogic_to_discrete(grid_signal, BREAKS)


# ------------ Forest Fire Cellular Automaton


class WindyForestFire(Operator):
    is_composition = False

    def __init__(self, grid_space=None, action_space=None, context_space=None):

        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=WIND_TYPE)

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action=None, context=None):
        return UPDATE(grid), context


# ------------ Utils


def get_failed_propagations_mask():
    """
    Here goes the only sampling of the step.
    """
    uniform_space = spaces.Box(low=0.0, high=1.0, shape=(ROW_K, COL_K), dtype=WIND_TYPE)
    uniform_roll = uniform_space.sample()
    
    failed_propagations = np.repeat(False, ROW_K * COL_K).reshape(ROW_K, COL_K)
    failed_propagations = WIND <= uniform_roll
    
    return failed_propagations


def get_kernel(failed_propagations):
    kernel = np.repeat(PROPAGATION, ROW_K * COL_K).reshape(ROW_K, COL_K)
    
    kernel[failed_propagations] = EMPTY
    kernel[1, 1] = IDENTITY
    
    return kernel


def convolve(grid, kernel):
    return convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=EMPTY)
    

def translate_analogic_to_discrete(grid, breaks):
    row, col = grid.shape
    
    # Init on empty by default
    empty = np.array(EMPTY, dtype=CELL_TYPE)
    new_grid = np.repeat(empty, row*col).reshape(row, col)
    
    # 4 Actions to carry out:
    
    # 1. Do nothing on EMPTYs
    # Implicitly defined by default values
    # empty_mask = grid < breaks.keep_break
    # new_grid[empty_mask] = EMPTY

    # 2. Keep some TREEs
    keep_mask = np.logical_and(grid >= breaks.keep_break, grid < breaks.burn_break)
    new_grid[keep_mask] = TREE
    
    # 3. Burn the remaining TREEs
    burn_mask = np.logical_and(grid >= breaks.burn_break, grid < breaks.consume_break)
    new_grid[burn_mask] = FIRE
    
    # 4. Consume the FIREs
    consume_mask = grid >= breaks.consume_break
    new_grid[consume_mask] = EMPTY    

    return new_grid
