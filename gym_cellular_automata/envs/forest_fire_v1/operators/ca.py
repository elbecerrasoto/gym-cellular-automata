import numpy as np
from gym import spaces

from gym_cellular_automata import Operator

# from ..utils.neighbors import neighborhood_at
# from ..utils.config import CONFIG

# Inwards semantics (towards tree)
WIND = [[0.12, 0.24, 0.48],
        [0.06, 0.00, 0.24],
        [0.06, 0.06, 0.12]]

WIND = np.array(WIND, dtype = np.float64)

uniform_space = spaces.Box(low=0.0, high=1.0, shape=(3, 3), dtype=np.float64)
uniform_roll = uniform_space.sample()

# propagation_success
# masked_wind > sampled


neighbors_ = (0, 0, 2,
              1, 1, 2,
              1, 1, 2) 

EMPTY = 0
TREE = 1
FIRE = 2

neighbors = np.array(neighbors_, dtype=np.uint8).reshape(3, 3)

active_wind = WIND.copy()

active_wind[neighbors == FIRE] *= 0.0

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
                    
                    new_grid[row][col] = self.fire

                elif cell == self.fire:
                    # Consume fire at each step
                    new_grid[row][col] = self.empty

                else:
                    continue

        return new_grid, context






    Neighbors = namedtuple(
        "Neighbors",
        [
            "up_left",
            "up",
            "up_right",
            "left",
            "self",
            "right",
            "down_left",
            "down",
            "down_right",
        ],
    )

    return Neighbors(
        up_left, up, up_right, left, self, right, down_left, down, down_right,
    )




strike = np.random.choice([True, False], 1, p=[p_fire, 1 - p_fire])[
    0
]
new_grid[row][col] = self.fire if strike else cell

# why the underscore though
# np.bool_
type(np.random.choice([True, False], p=[0.8, 0.2]))

def propagation_prob(neighbors):
    for neighbor in neighbors:
        

