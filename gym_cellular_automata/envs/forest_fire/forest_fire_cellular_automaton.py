import numpy as np
from gym import spaces

from gym_cellular_automata.builder_tools.data import Grid

from gym_cellular_automata.builder_tools.operators import CellularAutomaton, Modifier, Coordinator
from gym_cellular_automata.utils.neighbors import neighborhood_at, are_neighbors_a_boundary

# ------------ Forest Fire Cellular Automaton

CELL_SYMBOLS = {
    'empty': 0,
    'tree': 1,
    'fire': 2
    }

class ForestFireCellularAutomaton(CellularAutomaton):
    empty = CELL_SYMBOLS['empty']
    tree = CELL_SYMBOLS['tree']
    fire = CELL_SYMBOLS['fire']
    
    def __init__(self, grid_space):
        
        self.grid_space = grid_space
        self.action_space = None
        self.context_space = spaces.Box(0.0, 1.0, shape=(2,))

    def update(self, grid, action, context):
        # For the sequential update of a CA
        new_grid = Grid(grid.data.copy(), cell_states=3)
        p_fire, p_tree = context.data
        
        for row, cells in enumerate(grid.data):
            for col, cell in enumerate(cells):
                
                neighbors = neighborhood_at(grid, pos=(row, col), invariant=self.empty)
                
                if cell == self.tree and self.fire in neighbors:
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire
                
                elif cell == self.tree:
                    # Sample for lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1-p_fire])[0]
                    new_grid[row][col] = self.fire if strike else cell
                
                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1-p_tree])[0]
                    new_grid[row][col] = self.tree if growth else cell
                
                elif cell == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty
                
                else:
                    continue
                   
        return new_grid, context