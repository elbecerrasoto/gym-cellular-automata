from functools import reduce
from operator import mul
from collections import namedtuple

import numpy as np
from scipy.signal import convolve2d
from gym import spaces

from gym_cellular_automata import Operator

"""
+ $e < t < f \in \mathbb{R}_+$
+ $I, P \in \mathbb{R}_+$
+ $I >> P$

> $eI + 8fP < tI < tI + 8tP < tI + fP < tI + 8tP < fI$

1. $tI + 8tP < tI + fP$
2. $8tP < fP$
3. $8t < f$
"""


"""
### Advantages
0. Analogic world tricks (voting and stuff)
1. It gets rid of annoying get_neighbors utilities
2. It is fast
3. GPU?
4. Piggybacking on robust libraries
5. Deeper Understanding of Forest Fire

### Disadvantages
1. Complexity (coding, intuition)
2. Probs are fixed on each 1 step Update (just 1 sampling event)
3. Discretization gets difficult with many states and complex update logic
4. Complexity can be turn down by adding filters, still this probably is faster
    than an idented double for loop over python data structures.
"""

# ------------ Globals


# Cell Values
EMPTY = 0
TREE = 1
FIRE = 10

# Test Grid Size
ROW = 8
COL = 8

# Signal Weights
BASE = 2

I_EXP = 10
P_EXP = 3

IDENTITY = BASE ** I_EXP
PROPAGATION = BASE ** P_EXP


def assert_weights_relationships():
    assert EMPTY*IDENTITY + 8*FIRE*PROPAGATION < TREE*IDENTITY, "empty / tree"
    assert TREE*IDENTITY < TREE*IDENTITY + 8*TREE*PROPAGATION, "keep / keep"
    assert TREE*IDENTITY + 8*TREE*PROPAGATION < TREE*IDENTITY + FIRE*PROPAGATION, "keep / burn"
    assert TREE*IDENTITY + FIRE*PROPAGATION < TREE*IDENTITY + 8*FIRE*PROPAGATION, "burn / burn"
    assert TREE*IDENTITY + 8*FIRE*PROPAGATION < FIRE*IDENTITY, "burn / consume"


assert_weights_relationships()


# fmt: off
# Inwards semantics (towards tree)
WIND = [[1.00, 1.00, 1.00],
        [1.00, 0.00, 1.00],
        [1.00, 1.00, 1.00]]
# fmt: on

WIND = np.array(WIND, dtype=np.float64)

# Kernel Size
ROW_K = 3
COL_K = 3


# ------------ Breaks


def get_breaks():
    """ 3 breaks needed for 4 actions.
    empty < keep < burn < consume
    """
    keep_break = TREE * IDENTITY
    burn_break = TREE * IDENTITY + FIRE * PROPAGATION
    consume_break = FIRE * IDENTITY

    Breaks = namedtuple("Breaks", ["keep_break", "burn_break", "consume_break"])

    return Breaks(keep_break, burn_break, consume_break)

BREAKS = get_breaks()

# ------------ Utils


def get_failed_propagations_mask():
    """
    Here goes the only sampling of the step.
    """
    uniform_space = spaces.Box(low=0.0, high=1.0, shape=(ROW_K, COL_K), dtype=np.float64)
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
    

def get_breaks():
    """ 3 breaks needed for 4 actions.
    empty < keep < burn < consume
    """
    keep_break = TREE * IDENTITY
    burn_break = TREE * IDENTITY + FIRE * PROPAGATION
    consume_break = FIRE * IDENTITY

    Breaks = namedtuple("Breaks", ["keep_break", "burn_break", "consume_break"])

    return Breaks(keep_break, burn_break, consume_break)


def translate_analogic_to_discrete(grid, breaks):
    # Init on empty by default
    empty = np.array(EMPTY, dtype=np.uint8)
    new_grid = np.repeat(empty, ROW*COL).reshape(ROW, COL)
    
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


# ------------ 1-step CA update

def update(grid):
    # Sample Wind
    fail_to_propagate = get_failed_propagations_mask()
    
    kernel = get_kernel(fail_to_propagate)

    grid_signal = convolve(grid, kernel)

    return translate_analogic_to_discrete(grid_signal, BREAKS)


# ------------ Sample Run


def get_random_grid(shape=(ROW, COL), probs=[0.20, 0.70, 0.10], dtype=np.uint8):
    cell_values = np.array([EMPTY, TREE, FIRE], dtype=dtype)
    size = reduce(mul, shape)
    
    return np.random.choice(cell_values, size=size, p=probs).reshape(shape)


def main():

    grid = get_random_grid()
    
    for i in range(12):
        print(f'Grid at time {i}\n{grid}\n\n')
        grid = update(grid)


main()

# ------------ Forest Fire Cellular Automaton


class ForestFireCellularAutomaton(Operator):
    is_composition = False

    empty = CONFIG["cell_symbols"]["empty"]
    tree = CONFIG["cell_symbols"]["tree"]
    fire = CONFIG["cell_symbols"]["fire"]

    def __init__(self, grid_space=None, action_space=None, context_space=None):

        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = grid.copy()
        p_fire, p_tree = context

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):

                neighbors = neighborhood_at(grid, pos=(row, col), invariant=self.empty)

                if cell == self.tree:
                    # Which fires succesfully propagates.
                    pass

                elif cell == self.fire:
                    # Consume fire at each step
                    new_grid[row][col] = self.empty

                else:
                    continue

        return new_grid, context
