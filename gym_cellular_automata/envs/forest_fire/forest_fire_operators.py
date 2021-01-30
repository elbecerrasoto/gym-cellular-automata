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

# ------------ Forest Fire Modifier

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

# ------------ Forest Fire Coordinator

class ForestFireCoordinator(Coordinator):
    
    def __init__(self, cellular_automaton, modifier, steps_without_CA):
        
        self.suboperators = cellular_automaton, modifier
        self.cellular_automaton = cellular_automaton
        self.modifier = modifier
        
        self.steps_without_CA = steps_without_CA
        self.steps_without_CA_space = spaces.Discrete(steps_without_CA + 1)
        
        self.grid_space = cellular_automaton.grid_space
        
        self.action_space = modifier.action_space
        
        self.CA_params_space = cellular_automaton.context_space 
        self.pos_space = modifier.context_space
        self.context_space = spaces.Tuple((self.CA_params_space,
                                           self.pos_space,
                                           self.steps_without_CA_space))
        
    def update(self, grid, action, context):
        CA_params, pos, steps2CA = context
        
        if steps2CA == 0:
            
                grid, _ = self.cellular_automaton(grid, action, CA_params)
                grid, new_pos = self.modifier(grid, action, pos)
                
                steps2CA = self.steps_without_CA
            
        else:
            
            grid, new_pos = self.modifier(grid, action, pos)
            self.steps2CA -= 1
        
        context = CA_params, new_pos, steps2CA
        return grid, context
