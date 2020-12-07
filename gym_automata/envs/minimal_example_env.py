import gym
from gym import spaces

# Data Classes
from gym_automata.interface import Grid, ModifierState
# Service Classes
from gym_automata.interface import Automaton, Modifier, CAEnv

# Initialize Data Objects
# e.g. grid of 8x8 with 2 cell states
grid = 
custom_grid = 

# Test space membership

# Initialize Service Objects

class MinimalExampleEnv(gym.Env, CAEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, my_args):
    pass

  def step(self, action):
    # Call any number of updates over the grid
    # syntax: grid_operator.update(grid, action, modifier_state)
    # As many times as you want, with any logic whatsoever      
    def grid_operations(grid, action, modifier_state):
        grid = self.modifier.update(grid, action, modifier_state)
        grid = self.automaton.update(grid, action=None, modifier_state=None)
        return grid
    self.grid = grid_operations(self.grid, action, self.modifier_state)
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
    