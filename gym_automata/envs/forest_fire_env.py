import numpy as np
from collections import defaultdict

import gym
from gym import spaces, logger
from gym.utils import seeding

from gym_automata.interface import Grid, MoState
from gym_automata.interface import Automaton, Modifier
from gym_automata.interface import CAEnv

# ---------------- Globals

ROW = 5
COL = 5

CELL_STATES = 3
CELL_SYMBOLS = {
    'empty': 0,
    'tree': 1,
    'fire': 2
    }

P_FIRE = 0.033
P_TREE = 0.333

ACTIONS = 9
# Exchange of `fire` to `empty`
EFFECTS = {2: 0}
FREEZE = 2

REWARD_PER_EMPTY = 0.0
REWARD_PER_TREE = 1.0
REWARD_PER_FIRE = -1.0

# ---------------- Initial Data

def init_FF_grid(row, col, cell_states = 3):
    return Grid(shape = (row, col), cell_states = cell_states)

def init_FF_mostate(row, col, freeze):
    # Position of Helicopter
    hpos_space = spaces.MultiDiscrete([row, col])
    return MoState(mostate_space = hpos_space)

FF_GRID = init_FF_grid(ROW, COL, CELL_STATES)
FF_MOSTATE = init_FF_mostate(ROW, COL, FREEZE)

# ---------------- Spaces

FF_GRID_SPACE = FF_GRID.grid_space
FF_MOSTATE_SPACE = FF_MOSTATE.mostate_space

FF_ACTION_SPACE = spaces.Discrete(ACTIONS)
FF_OBSERVATION_SPACE = spaces.Tuple((FF_GRID_SPACE, FF_MOSTATE_SPACE))

# ---------------- Env

class ForestFireCAEnv(CAEnv, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.freeze = FREEZE
        
        # Data
        self.initial_grid = FF_GRID
        self.initial_mostate = FF_MOSTATE

        self.n_row = FF_GRID.data.shape[0]
        self.n_col = FF_GRID.data.shape[1]

        self.cell_symbols = CELL_SYMBOLS
        self.empty = CELL_SYMBOLS['empty']
        self.tree = CELL_SYMBOLS['tree']
        self.fire = CELL_SYMBOLS['empty']
        
        self.reward_per_empty = REWARD_PER_EMPTY
        self.reward_per_tree = REWARD_PER_TREE
        self.reward_per_fire = REWARD_PER_FIRE
        
        # Operators
        self.automaton = ForestFire(P_FIRE, P_FIRE, CELL_SYMBOLS, FF_GRID_SPACE)
        self.modifier = Helicopter(EFFECTS, FF_GRID_SPACE, FF_MOSTATE_SPACE, FF_ACTION_SPACE)
        
        # Spaces
        # Data Spaces
        self.grid_space = FF_GRID_SPACE
        self.mostate_space = FF_MOSTATE_SPACE    
        # RL Spaces
        self.observation_space = FF_OBSERVATION_SPACE
        self.action_space = FF_ACTION_SPACE

    def step(self, action):
        done = self._is_terminated()
        if not done:
            
            if self.steps2CA == 0:
                
                grid = self.automaton.update(self.grid)
                self.mostate = self._move_helicopter(action)
                self.grid = self.modifier.update(grid, action, self.mostate)
                self.steps2CA = self.freeze
            
            else:
                
                self.grid = self.modifier.update(self.grid, action, self.mostate)
                self.steps2CA -= 1
                
            self.steps += 1
            self.hits += self.modifier.hit
            info = {'steps': self.steps, 'hits': self.hits}
            reward = self._calculate_reward()
            obs = self.grid.data, self.mostate.data
            
            return obs, reward, done, info
        
        else:
            logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            
            return (self.grid.data, self.mostate.data), 0.0, True, {'steps': self.steps, 'hits': self.hits}

    def reset(self):
        # If random start, uncomment
        # self.initial_grid.data = self.initial_grid.grid_space.sample()
        # self.initial_mostate.data = self.initial_mostate.mostate_space.sample()
        
        self.grid = self.initial_grid
        self.mostate = self.initial_mostate
        
        self.steps2CA = self.freeze

        self.steps = 0
        self.hits = 0
        
        obs = self.grid, self.mostate
        reward = self._calculate_reward()
        done = self._is_terminated()
        info = {'steps': self.steps, 'hits': self.hits}
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        pass

    def close(self):
        print('Gracefully closing forest fire environment.')
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_reward(self):
        cell_types, counts = np.unique(self.grid.data, return_counts=True)
        cell_counts = defaultdict(int, zip(cell_types, counts))

        weights = np.array([self.reward_per_empty, self.reward_per_tree, self.reward_per_fire])
        counts = np.array([cell_counts[self.empty], cell_counts[self.tree], cell_counts[self.fire]])

        return np.dot(weights, counts)
    
    def _is_terminated(self, mode='continuing'):
        return False
    
    def _move_helicopter(self, action):
        row = self.mostate.data[0]
        col = self.mostate.data[1]
        
        def is_out_borders(action, pos):    
            if pos == 'row':
                # Check Up movement
                if action in {1, 2, 3} and row == 0:
                    out_of_border = True
                # Check Down movement
                elif action in {7, 8, 9} and row == self.n_row-1:
                    out_of_border = True
                else:
                    out_of_border = False
            
            elif pos == 'col':
                # Check Left movement
                if action in {1, 4, 7} and col == 0:
                    out_of_border = True
                # Check Right movement
                elif action in {3, 6, 9} and col == self.n_col-1:
                    out_of_border = True
                else:
                    out_of_border = False
            else:
                raise ValueError('invalid argument: pos = "row" | "col"')            
            return out_of_border
        
        def new_hpos(action, row, col):
            assert self.action_space.contains(action), f'action: {action} does not belong to {self.action_space}'
            
            new_row = row if is_out_borders(action, pos='row')\
                else row if action in {5}\
                else row - 1 if action in {1, 2, 3}\
                else row + 1 if action in {7, 8, 9}\
                else None
            
            new_col = col if is_out_borders(action, pos='col')\
                else col if action in {5}\
                else col - 1 if action in {1, 4, 7}\
                else col + 1 if action in {3, 6, 9}\
                else None
            
            assert not(new_row is None or new_col is None), 'fatal error: `hpos` cannot be `None`.'
            return np.array([new_row, new_col])
        
        return new_hpos(action, row, col)

# ---------------- Operator Objects

class ForestFire(Automaton):
    
    def __init__(self, p_fire, p_tree, cell_symbols, grid_space):
        self.p_fire = p_fire
        self.p_tree = p_tree
        self.cell_symbols = cell_symbols   
        
        self.grid_space = grid_space
    
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
        up_right = grid[row-1, col+1] if legality['up'] and legality['right'] else invariant

        middle_left = grid[row, col-1] if legality['left'] else invariant
        middle = grid[row, col]
        middle_right = grid[row, col+1] if legality['right'] else invariant
        
        down_left = grid[row+1, col-1] if legality['down'] and legality['left'] else invariant
        down_center = grid[row+1, col] if legality['down'] else invariant
        down_right = grid[row+1, col+1] if legality['down'] and legality['right'] else invariant

        return up_left, up_center, up_right, middle_left, middle, middle_right, down_left, down_center, down_right

class Helicopter(Modifier):

    def __init__(self, effects, grid_space, mostate_space, action_space):
        self.effects = effects
        
        # Spaces
        self.grid_space = grid_space
        self.mostate_space = mostate_space
        self.action_space = action_space

    def update(self, grid, action, mostate):
        row, col = mostate.data
        self.hit = False
        # Exchange of cells if applicable (usually `fire` to `empty`).
        for symbol in self.effects:
            if grid[row, col] == symbol:
                grid[row, col] = self.effects[symbol]
                self.hit = True
        return grid
