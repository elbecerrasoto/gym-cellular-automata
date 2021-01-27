import numpy as np
from collections import namedtuple

from gym_cellular_automata.builder_tools.operators import CellularAutomaton, Modifier, Coordinator

def are_neighbors_a_boundary(grid, pos):
    """
    Check if the neighbors of target position are a boundary.
    Return a tuple of Bools informing which neighbor is a boundary.
    It checks the up, down, left, and right neighbors.
    """
    row, col = pos 
    n_row, n_col = grid.data.shape
    
    up_offset, down_offset = row + np.array([-1, 1])
    left_offset, right_offset = col + np.array([-1, 1])

    up = up_offset >= 0
    down = down_offset <= n_row-1
    left = left_offset >= 0
    right = right_offset <= n_col-1
    
    LegalBounds = namedtuple('Bounds', ['up', 'down', 'left', 'right'])
    return LegalBounds(up, down, left, right)

# ------------ Forest Fire Cellular Automaton

class ForestFireCellularAutomaton(CellularAutomaton):
    
    def __init__(self, cell_symbols, grid_space):
        self.cell_symbols = cell_symbols   
        
        self.grid_space = grid_space
        self.action_space = None
        self.context_space = spaces.Box

    def update(self, grid, action, context):
        new_data = grid.data.copy() # For the sequential update of a CA
        p_fire, p_tree = context.data
        
        empty = self.cell_symbols['empty']
        tree = self.cell_symbols['tree']
        fire = self.cell_symbols['fire']

        def is_fire_around(grid, pos):
            row, col = pos
            
            fire_around = False
            for neightbor in self._neighborhood(grid, pos):
                if neightbor == fire:
                    fire_around = True
                    break
            return fire_around
    
        for row, cells in enumerate(grid.data):
            for col, cell in enumerate(cells):
                
                if cell == tree and is_fire_around(grid, pos=(row, col)):
                    # Burn tree to the ground
                    new_data[row][col] = fire
                
                elif cell == tree:
                    # Sample for lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1-p_fire])[0]
                    new_data[row][col] = fire if strike else cell
                
                elif cell == empty:
                    # Sample to grow a tree
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1-p_tree])[0]
                    new_data[row][col] = tree if growth else cell
                
                elif cell == fire:
                    # Consume fire
                    new_data[row][col] = empty
                
                else:
                    continue
        
        grid.data = new_data                
        return grid, context

    def _neighborhood(self, grid, pos):
        """
        Calculates the Moore's neighborgood of cell at target position `pos`.
        The boundary conditions are invariant and set to `empty`.
        Returns a tuple with the values of the nighborhood cells in the following
        order: up_left, up_center, up_right,
                middle_left, middle, middle_right,
                down_left, down_center, down_right
        """        
        invariant = self.cell_symbols['empty']
        row, col = pos

        legality = are_neighbors_a_boundary(grid, pos)   

        up_left = grid[row-1, col-1] if legality.up and legality.left else invariant
        up_center = grid[row-1, col] if legality.up else invariant       
        up_right = grid[row-1, col+1] if legality.up and legality.right else invariant

        middle_left = grid[row, col-1] if legality.left else invariant
        middle = grid[row, col]
        middle_right = grid[row, col+1] if legality.right else invariant
        
        down_left = grid[row+1, col-1] if legality.down and legality.left else invariant
        down_center = grid[row+1, col] if legality.down else invariant
        down_right = grid[row+1, col+1] if legality.down and legality.right else invariant

        return up_left, up_center, up_right, middle_left, middle, middle_right, down_left, down_center, down_right

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
    
# context_space = (((p_fire, p_tree)), ((row, col)), freeze)

# I need an empty space, contast space.

# assume an specific context
# new_context?
# new_action?

# If you do not want to idented
# That's why is separeted
# CoordinatorContext=((row, col), freeze)


# If you do not want to compute from the suboperators the spaces
# def __init__(self, cellular_automaton, modifier, grid_space, action_space, context_space):

# steps_to_CA_update on global env

class ForestFireCoordinator(Coordinator):
    
    def __init__(self, cellular_automaton, modifier, steps_without_CA):
        
        self.suboperators = (cellular_automaton, modifier)
        self.cellular_automaton = cellular_automaton
        self.modifier = modifier
        
        self.grid_space = cellular_automaton.grid_space
        self.action_space = modifier.action_space
        self.context_space = 
    
    def update(self, grid, action, context):
        pass