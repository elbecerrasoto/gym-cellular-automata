import numpy as np
from gymnasium import spaces

from gym_cellular_automata.forest_fire.utils.neighbors import neighborhood_at
from gym_cellular_automata.operator import Operator


def normalize_p(p):
    p = np.asarray(p).astype("float64")
    p = p / np.sum(p)
    return p


class ForestFire(Operator):
    grid_dependant = True
    action_dependant = False
    context_dependant = True

    deterministic = False

    def __init__(self, empty, tree, fire, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.empty = empty
        self.tree = tree
        self.fire = fire

        if self.context_space is None:
            self.context_space = spaces.Box(0.0, 1.0, shape=(2,))

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = grid.copy()
        p_fire, p_tree = context

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):
                neighbors = neighborhood_at(grid, (row, col), invariant=self.empty)

                if cell == self.tree and self.fire in neighbors:
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire

                elif cell == self.tree:
                    # Sample for lightning strike

                    strike = self.np_random.choice(
                        [True, False], p=normalize_p([p_fire, 1 - p_fire])
                    )

                    new_grid[row][col] = self.fire if strike else cell

                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = self.np_random.choice(
                        [True, False], p=normalize_p([p_tree, 1 - p_tree])
                    )

                    new_grid[row][col] = self.tree if growth else cell

                elif cell == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty

        return new_grid, context
