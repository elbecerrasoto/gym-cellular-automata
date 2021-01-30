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

class ForestFireModifier(Modifier):
    hit = False    
    
    def __init__(self, effects, grid_space, action_space, context_space):
        self.effects = effects
        
        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        pos = context.data
        new_pos = self._move(action, pos)
        
        self.hit = False
        # Exchange of cells if applicable (usually `fire` to `empty`).
        for symbol in self.effects:
            if grid[new_pos] == symbol:
                grid[new_pos] = self.effects[symbol]
                self.hit = True

        context.data = new_pos
        return grid, context

    def _move(self, grid, action, pos):
        assert self.action_space.contains(action), f'action: {action} does not belong to {self.action_space}'
        
        row, col = pos
        legality = are_neighbors_a_boundary(grid, pos)
        
        new_row = row if legality.up and legality.down\
            else row if action in {5}\
            else row - 1 if action in {1, 2, 3}\
            else row + 1 if action in {7, 8, 9}\
            else None
        
        new_col = col if legality.left and legality.right\
            else col if action in {5}\
            else col - 1 if action in {1, 4, 7}\
            else col + 1 if action in {3, 6, 9}\
            else None
        
        assert not(new_row is None or new_col is None), '`new_row` or `new_col` cannot be `None`.'
        return np.array([new_row, new_col])
