import numpy as np

import gym
from gym import spaces

from gym_automata.interface import Grid, MoState
from gym_automata.interface import Automaton, Modifier
from gym_automata.interface import CAEnv

# ---------------- Initialize Data Objects

ROW = 5
COL = 5

P_FIRE = 0.033
P_TREE = 0.333

FREEZE = 2 

CELL_SYMBOLS = {
    'empty': 0,
    'tree': 1,
    'fire': 2
    }

# ---------------- Grid Object

# A 5x5 CA forest fire grid.
FF_GRID = Grid(shape = (ROW,COL), cell_states = 3)

# ---------------- MoState Object

# Position of Helicopter
hpos_space = spaces.MultiDiscrete( [ROW,COL] )
# CA freeze steps
freeze_space = spaces.Discrete( FREEZE )
# Combined spaces
tspaces = hpos_space, freeze_space
hpos_freeze_space = spaces.Tuple(tspaces)

# Init the modifier
FF_MOSTATE = MoState(mostate_space =  hpos_freeze_space)

# ---------------- Code Operator Objects

class ForestFire(Automaton):
    grid_space = FF_GRID.grid_space
    
    def __init__(self, p_fire, p_tree, cell_symbols):
        self.p_fire = p_fire
        self.p_tree = p_tree
        self.cell_symbols = cell_symbols        
    
    def update(self, grid, action=None, mostate=None):
        new_data = grid.data.copy()
        
        empty = self.cell_symbols['empty']
        tree = self.cell_symbols['tree']
        fire = self.cell_symbols['fire']
        p_fire = self.p_fire
        p_tree = self.p_tree
        
        def is_fire_around(grid, row, col):
            fire_around = False
            for neightbor in self._neighborhood(grid, row, col):
                if neightbor == fire:
                    fire_around = True
                    break
            return fire_around
    
        for row, cells in enumerate(grid.data):
            for col, cell in enumerate(cells):
                
                if cell == tree and is_fire_around(grid, row, col):
                    # Burn tree to the ground
                    new_data[row][col] = fire
                
                elif cell == tree:
                    # Roll a dice for a lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1-p_fire])[0]
                    new_data[row][col] = fire if strike else cell
                
                elif cell == empty:
                    # Roll a dice for a growing bush
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1-p_tree])[0]
                    new_data[row][col] = tree if growth else cell
                
                elif cell == fire:
                    # Consume fire
                    new_data[row][col] = empty
                
                else:
                    continue
        
        grid.data = new_data                
        return grid

    def _neighborhood(self, grid, row, col):
        """
        Calculates the Moore's neighborgood of cell at `row`, `col`.
        The boundary conditions are invariant and set to 'empty'.
        Returns a tuple with the values of the nighborhood cells in the following
        order: up_left, up_center, up_right,
                middle_left, middle, middle_right,
                down_left, down_center, down_right
        """        
        invariant = self.cell_symbols['empty']
        
        def are_bounds_legal(grid, row, col):
            """
            Check borders on target cell
            """
            n_row = grid.data.shape[0]
            n_col = grid.data.shape[1]
            r_offset = row + np.array([-1, 1])
            c_offset = col + np.array([-1, 1])
            up = r_offset[0] >= 0
            down = r_offset[1] <= n_row-1
            left = c_offset[0] >= 0
            right = c_offset[1] <= n_col-1
            return {'up': up, 'down': down, 'left': left, 'right': right}

        legality = are_bounds_legal(grid, row, col)   

        up_left = grid[row-1, col-1] if legality['up'] and legality['left'] else invariant
        up_center = grid[row-1, col] if legality['up'] else invariant       
        up_right = grid[row-1, col+1]if legality['up'] and legality['right'] else invariant

        middle_left = grid[row, col-1] if legality['left'] else invariant
        middle = grid[row, col]
        middle_right = grid[row, col+1] if legality['right'] else invariant
        
        down_left = grid[row+1, col-1] if legality['down'] and legality['left'] else invariant
        down_center = grid[row+1, col] if legality['down'] else invariant
        down_right = grid[row+1, col+1] if legality['down'] and legality['right'] else invariant

        return up_left, up_center, up_right, middle_left, middle, middle_right, down_left, down_center, down_right

FF_AUTOMATON = ForestFire(P_FIRE, P_TREE, CELL_SYMBOLS)

FF_GRID.data = FF_GRID.grid_space.sample()
FF_GRID
FF_AUTOMATON.update(FF_GRID)

for i in range(10):
    FF_AUTOMATON.update(FF_GRID)
    print(FF_GRID)


grid_x = Grid(shape=(23,10), cell_states=3)
FF_AUTOMATON.update(grid_x)
