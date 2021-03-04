import numpy as np
from gym import spaces

from gym_cellular_automata import Operator

# from ..utils.neighbors import neighborhood_at
# from ..utils.config import CONFIG

# Inwards semantics (towards tree)

EMPTY = 0
TREE = 1
FIRE = 10


WIND = [[1.00, 0.60, 1.00], [0.06, 0.00, 0.24], [0.06, 0.06, 0.12]]

WIND = np.array(WIND, dtype=np.float64)

uniform_space = spaces.Box(low=0.0, high=1.0, shape=(3, 3), dtype=np.float64)


# ------------ Sample just one per update


uniform_roll = uniform_space.sample()

failed_propagations = np.repeat(False, 3 * 3).reshape(3, 3)
failed_propagations = WIND <= uniform_roll


# ------------ Get the kernel


IDENTITY = 2 ** 10
PROPAGATION = 2 ** 5

tree_break = TREE * IDENTITY
fire_break = FIRE * IDENTITY

non_propagation = TREE * IDENTITY + 8 * TREE * PROPAGATION
propagate = TREE * IDENTITY + FIRE * PROPAGATION

breaks = tree_break, non_propagation, propagate, fire_break

# changing the kernel on each update
# First get the kernel
# pytorch stuff is pretty easy

ROW_KERNEL = 3
COL_KERNEL = 3

kernel = np.repeat(PROPAGATION, 3 * 3).reshape(3, 3)


kernel[failed_propagations] = 0

kernel[1, 1] = IDENTITY


# ------------ Test on a random grid


cell_values = np.array([EMPTY, TREE, FIRE], dtype=np.uint8)

ROW = 8
COL = 8
random_grid = np.random.choice(
    cell_values, size=ROW * COL, p=[0.20, 0.70, 0.10]
).reshape(ROW, COL)


random_grid


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
