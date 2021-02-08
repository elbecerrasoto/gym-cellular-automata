import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from ..utils.neighbors import are_my_neighbors_a_boundary

# ------------ Forest Fire Modifier

class ForestFireModifier(Operator):
    is_composition = False
    hit = False

    def __init__(self, effects, grid_space=None, action_space=None, context_space=None):
        
        self.effects = effects
        
        if action_space is None:
            action_space = spaces.Box(1, 9, shape=tuple(), dtype=np.uint8)
        
        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        new_pos = self._move(grid, action, pos=context)
        row, col = new_pos
        
        self.hit = False
        
        # If applicable, cell state changes from FIRE to EMPTY.
        for symbol in self.effects:
            
            if grid[row, col] == symbol:      
                grid[row, col] = self.effects[symbol]
                self.hit = True

        return grid, new_pos

    def _move(self, grid, action, pos):
        action = np.array(action)
        
        if not self.action_space.contains(action):
            raise ValueError(f'action: {action} does not belong to {self.action_space}')
        
        row, col = pos

        is_boundary = are_my_neighbors_a_boundary(grid, pos)
        
        new_row = row - 1 if not is_boundary.up    and int(action) in {1, 2, 3} else \
                  row + 1 if not is_boundary.down  and int(action) in {7, 8, 9} else \
                  row
        
        new_col = col - 1 if not is_boundary.left  and int(action) in {1, 4, 7} else \
                  col + 1 if not is_boundary.right and int(action) in {3, 6, 9} else \
                  col

        return np.array([new_row, new_col])
