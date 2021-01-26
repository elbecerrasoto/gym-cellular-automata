import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym_automata.interface.data import Grid, State
from gym_automata.interface.operators import Automaton, Exchanger, Synchronizer

SHAPE = (2, 2)

# ---------------- Automaton

class MinimalCAEnvAutomaton(Automaton):
    
    def __init__(self, grid_space, action_space, state_space):
        self.grid_space = grid_space
        self.action_space = action_space
        self.state_space = state_space

    def update(self, grid, action, state):
        grid_copy = grid.data.copy()
        
        def apply_CA_local_rule(data, action, state):
            return data

        grid.data = apply_CA_local_rule(grid_copy)
        return grid


# ---------------- Exchanger

class MinimalCAEnvExchanger(Exchanger):
    
    def __init__(self, grid_space, action_space, state_space, effects):
        self.grid_space = grid_space
        self.action_space = action_space
        self.state_space = state_space
        
        self.effects = effects

    def update(self, grid, action, state):
        grid_copy = grid.data.copy()
        
        def should_exchange(current_cell):
            return np.random.choice((True, False))
        
        def exchange(grid_data, effects, rule)
            for row in grid_data.shape[0]:
                for col in grid_data.shape[1]:
                    for symbol in effects:
                        if rule(grid_data[row, col]):
                            grid_data[row, col] = effects[symbol]
                            
        grid.data = exchange(grid_copy, self.effects, should_exchange)
        return grid
            
        
        for symbol in self.effects:
            if grid[row, col] == symbol:
                grid[row, col] = self.effects[symbol]
                self.hit = True
        return grid
    
# ---------------- Synchronizer

class MinimalCAEnvSynchronizer():
    
    def __init__(self, automata, exchanger):
        
        # Operators
        self.automata = automata
        self.exchanger = exchanger
        
        # Spaces
        self.grid_space = grid_space
        self.action_space = action_space
        self.state_space = state_space
    
    def update(self, grid, action, state):
        if orst.data == 0:
            
            grid = self.automaton.update(grid, action=None, most=None, orst=None)            
            grid = self.modifier.update(grid, action, most, orst=None)
            
        else:
            grid = self.modifier.update(grid, action, most, orst=None)
        
        return grid
