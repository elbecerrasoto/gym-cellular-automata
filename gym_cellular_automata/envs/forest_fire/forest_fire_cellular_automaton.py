import numpy as np
from gym import spaces

from gym_cellular_automata import Grid

from gym_cellular_automata import Operator
from gym_cellular_automata.utils.neighbors import neighborhood_at

CONFIG_FILE = 'gym_cellular_automata/envs/forest_fire/forest_fire_config.yaml'

def get_config_dict(file):
    import yaml
    yaml_file = open(file, 'r')
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content

CONFIG = get_config_dict(CONFIG_FILE)

# ------------ Forest Fire Cellular Automaton

class ForestFireCellularAutomaton(Operator):
    empty = CONFIG['cell_symbols']['empty']
    tree = CONFIG['cell_symbols']['tree']
    fire = CONFIG['cell_symbols']['fire']
    
    def __init__(self, grid_space=None, action_space=None, context_space=None):
        
        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(2,))
        
        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = Grid(grid.copy(), cell_states=3)
        p_fire, p_tree = context
        
        for row, cells in enumerate(grid):
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
