import gym
from gym import spaces

# Data classes
from gym_automata.interface import Grid, MoState
# Operator classes
from gym_automata.interface import Automaton, Modifier
# Organizer classes
from gym_automata.interface import CAEnv

# ---------------- Initialize Data Objects

# ---------------- Grid Object

# A 8x8 Cellular Automaton Grid, with 2 cell states and random data.
grid = Grid(shape = (8,8), cell_states = 2)
# Access the data
# >>> grid.data
# >>> grid[:]

# ---------------- MoState Object

# Declare a space
# For example:s
# Discrete with 8 elements, from 0 to 7
mostate_space1 = spaces.Discrete(8)
# Continuos from 0-360, maybe they are angle degrees
mostate_space2 = spaces.Box(low=0, high=360, shape=tuple())
# Unbounded range, maybe this space represents coordinates
mostate_space3 = spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,2))

# Sample using the sample method
# >>> mostate_space3.sample()
# Test membership by contains
# >>> mostate_space1.contains(7)
# >>> mostate_space1.contains(256)

# Combine spaces with Tuple, to get a new combined space
tspaces = (mostate_space1, mostate_space2, mostate_space3)
mostate_space = spaces.Tuple(tspaces)

# MoState with random data
mostate = MoState(mostate_space = mostate_space)
# Access the data
# >>> mostate.data
# >>> mostate[:]

# ---------------- Code Operator Objects

class MyCellularAutomaton(Automaton):
    def __init__(self, grid_space):
        self.grid_space = grid_space

    def update(self, grid, action=None, mostate=None):
        return grid

class MyModifier(Modifier):
    def __init__(self, grid_space, action_space, mostate_space):
        self.grid_space = grid_space
        self.action_space = action_space
        self.mostate_space = mostate_space

    def update(self, grid, action, mostate):
        return grid

automaton = MyCellularAutomaton(grid.grid_space)

# Test update method
# >>> automaton.update(grid)

# Maybe going left or right
action_space = spaces.Discrete(2)
modifier = MyModifier(grid.grid_space, action_space, mostate.mostate_space)

# Test update method
action = modifier.action_space.sample()
modifier.update(grid, action, mostate)

# ---------------- Code your happy environment
# Coordinate the Automaton and Modifier operations into a coherent Gym Environment

class MinimalExampleEnv(CAEnv, gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        # Data
        self.grid = grid
        self.mostate = mostate
        
        # Services
        self.automaton = automaton
        self.modifier = modifier

        # Data Spaces
        self.grid_space = grid.grid_space
        self.mostate_space = mostate.mostate_space
      
        # RL Spaces
        self.action_space = modifier.action_space
        if isinstance(modifier.mostate_space, spaces.Space):
            self.observation_space = spaces.Tuple((grid.grid_space, modifier.mostate_space))
        else:
            self.observation_space = grid.grid_space

    def step(self, action):
        # Call any number of updates over the grid
        # syntax: grid_operator.update(grid, action, MoState)
        # As many times as you want, with any logic whatsoever      
        def grid_operations(grid, action, mostate):
            grid = self.modifier.update(grid, action, mostate)
            grid = self.automaton.update(grid, action=None, mostate=None)
            return grid
        self.grid = grid_operations(self.grid, action, self.mostate)
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

# Test MinimalExampleEnv
# >>> env = MinimalExampleEnv()
# >>> env.step(env.action_space.sample())
# >>> env.render()
# >>> env.reset()
# >>> env.close()
