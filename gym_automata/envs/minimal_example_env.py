import gym
from gym import spaces
import numpy as np
# Data Classes
from gym_automata.interface import Grid, State
# Service Classes
from gym_automata.interface import Automaton, Modifier, CAEnv

# ---------------- Initialize Data Objects

# ---------------- Grid Object
# A Grid data object needs four pieces of information
# 1. data (optional, random sampled if not provided)
# 2. shape (multidimensional shape, usually 2-D)
# 3. cell_states (n number of cell states, they will be labeled from 0 to n-1)
# 4. cell_type (optional, default=np.int32)

# e.g. grid of 8x8 with 2 cell states and random initialization
grid = Grid(shape=(8,8), cell_states=2)

# e.g. grid of 8x8 with 2 cell states and custom initialization
grid_ones = Grid(data=np.ones((8,8)), shape=(8,8), cell_states=2)

# Access the data as a numpy array
grid[:]
grid_ones[:]

# Check if both Grids lie on the same space
grid.grid_space.contains(grid_ones[:])
grid_ones.grid_space.contains(grid[:])

# ---------------- State Object
# A State data object needs two pieces of information
# 1. data
# 2. space (a Gym Space type)

# Declare a space
# e.g. A discrete space from 0 to 7
state_space = spaces.Discrete(8)

# State data must be explicitly provided
# So let's create a random sample
state_data = state_space.sample()

# Create a ModifierState data object
state = State(data=state_data, state_space=state_space)

# Access the data by its __call__() method
state()

# ---------------- Code Service Objects
# Implement the Service Objects
class MyCellularAutomaton(Automaton):
    def __init__(self, grid_space):
        self.grid_space = grid_space

    def update(self, grid, action=None, state=None):
        return grid

class MyModifier(Modifier):
    def __init__(self, grid_space, action_space, state_space):
        self.grid_space = grid_space
        self.action_space = action_space
        self.state_space = state_space

    def update(self, grid, action, state):
        return grid

automaton = MyCellularAutomaton(grid.grid_space)
automaton.update(grid)

modifier = MyModifier(grid.grid_space, spaces.Discrete(9), spaces.Discrete(8))
modifier.update(grid, modifier.action_space.sample(), modifier.state_space.sample())

# ---------------- Code your happy Environment
# Wrap the Automaton and Modifier into a coherent Gym Environment
# The Environment performs operations over a grid calling the method update from the Service Objects.
# After a series of grid operations
# an Observation, a Reward, a Termination Signal and an Information are returned
class MinimalExampleEnv(CAEnv, gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        # Data
        self.grid = grid
        self.state = state
        
        # Services
        self.automaton = automaton
        self.modifier = modifier

        # Data Spaces
        self.grid_space = grid.grid_space
        self.state_space = state.state_space
      
        # RL Spaces
        self.action_space = modifier.action_space
        if isinstance(modifier.state_space, spaces.Space):
            self.observation_space = spaces.Tuple((grid.grid_space, modifier.state_space))
        else:
            self.observation_space = grid.grid_space

    def step(self, action):
        # Call any number of updates over the grid
        # syntax: grid_operator.update(grid, action, state)
        # As many times as you want, with any logic whatsoever      
        def grid_operations(grid, action, state):
            grid = self.modifier.update(grid, action, state)
            grid = self.automaton.update(grid, action=None, state=None)
            return grid
        self.grid = grid_operations(self.grid, action, self.state)
        # Just don't forget to return an Observation, a Reward, a Termination, and an Info
        obs = self.observation_space.sample()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info
              
    def reset(self):
        print('Tic, tac, big bang!')
    
    def render(self, mode='human'):
        print('For educational purposes only.')
    
    def close(self):
        print('Hasta la vista baby!')

env = MinimalExampleEnv()
env.step(env.action_space.sample())
env.render()
env.reset()
env.close()
