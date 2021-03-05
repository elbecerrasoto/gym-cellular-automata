import numpy as np
from scipy.signal import convolve2d
from gym import spaces

from gym_cellular_automata import Operator

"""
+ $e < t < f \in \mathbb{R}_+$
+ $I, P \in \mathbb{R}_+$

> $eI + 8fP < tI < tI + 8tP < tI + fP < tI + 8tP < fI$

1. $tI + 8tP < tI + fP$
2. $8tP < fP$
3. $8t < f$
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
IDENTITY = 2 ** 10
PROPAGATION = 2 ** 5

# Inwards semantics (towards tree)
WIND = [[1.00, 1.00, 1.00],
        [1.00, 0.00, 1.00],
        [1.00, 1.00, 1.00]]
WIND = np.array(WIND, dtype=np.float64)

# Kernel Size
ROW_K = 3
COL_K = 3

# ------------ Run test

# ------------ Sampling


uniform_space = spaces.Box(low=0.0, high=1.0, shape=(3, 3), dtype=np.float64)
uniform_roll = uniform_space.sample()

failed_propagations = np.repeat(False, 3 * 3).reshape(3, 3)
failed_propagations = WIND <= uniform_roll

# ------------ Get kernel for 1-step update

# Breaks
tree_break = TREE * IDENTITY
fire_break = FIRE * IDENTITY

non_propagation = TREE * IDENTITY + 8 * TREE * PROPAGATION
propagate = TREE * IDENTITY + FIRE * PROPAGATION

kernel = np.repeat(PROPAGATION, ROW_K * COL_K).reshape(ROW_K, COL_K)
kernel[failed_propagations] = 0
kernel[1, 1] = IDENTITY


def get_kernel(failed_propagations):
    kernel = np.repeat(PROPAGATION, ROW_K * COL_K).reshape(ROW_K, COL_K)
    kernel[failed_propagations] = EMPTY
    kernel[1, 1] = IDENTITY
    return kernel


# ------------ Convs with scipy

convolved = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)

# Init on empty by default
empty = np.array(EMPTY, dtype=np.uint8)
new_grid = np.repeat(empty, ROW*COL).reshape(ROW, COL)

# Keep Tree
keep_tree = np.logical_and(convolved >= tree_break, convolved < non_propagation)

# Burn Tree
burn_tree = np.logical_and(convolved >= non_propagation, convolved < propagate)

# Consume Fire
consume_fire = convolved >= fire_break

# ------------ Translate

# Keep Tree
new_grid[keep_tree] = TREE

# Burn Tree
new_grid[burn_tree] = FIRE

# Consume Fire
new_grid[consume_fire] = EMPTY

# ------------ Manual Assess

grid

new_grid


# ------------ Random grid

cell_values = np.array([EMPTY, TREE, FIRE], dtype=np.uint8)

random_grid = np.random.choice(
    cell_values, size=ROW * COL, p=[0.20, 0.70, 0.10]
).reshape(ROW, COL)

grid = random_grid

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
