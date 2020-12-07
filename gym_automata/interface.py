#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:59 2020

@author: ebecerra
"""

import numpy as np
from gym import spaces

# -------- Data Classes --------
class Grid:
    """
    Doc
    """
    def __init__(self, data=None, shape=None, cell_states=None, cell_type=np.int32):
        assert not(shape is None or cell_states is None), 'shape and cell_states must be explicitly provided.'
        self.grid_space = spaces.Box(low=0, high=cell_states-1, shape=shape, dtype=cell_type)
        if data is None:
            self.data = self.grid_space.sample()
        elif self.grid_space.contains(data):
            self.data = data
        else:
            raise ValueError('Invalid grid data.')

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()

class ModifierState:
    """
    Doc
    """
    def __init__(self, data=None, space=None):
        self.modifier_state_space = space
        if data is None:
            self.data = data
        elif self.modifier_state_space.contains(data):
            self.data = data
        else:
            raise ValueError('Invalid modifier state data.')

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()

# -------- Service Classes --------
class Automaton:
    """
    It operates over a grid,
    performing an update by following the Automaton rules and neighborhood.
    """

    def update(self, grid, action=None, modifier_state=None):
        """Operation over a grid.
        
        Args:
            grid (object): a grid provided by the env
        Returns:
            grid (object): modified grid
        """
        raise NotImplementedError

class Modifier:
    """    
    It operates over a grid,
    represeting some sort of control over it.
    
    Its main method is update, which receives an action and a grid and returns
    a grid and a hidden state, both are used by the CAenv class to define an
    environment with the semantics for an Reinforcement Learning task.
    """
    # Set these in ALL subclasses
    modifier_state = None

    def update(self, grid, action, modifier_state):
        """Operation over a grid. It represents the control task
        to perform.
        
        Args:
            action (object): an action provided by the agent
            grid (object): a grid provided by the env
        Returns:
            grid (object): modified grid
            modifier_state (object): current modifier state
        """
        raise NotImplementedError

class CAEnv:
    """   
    It operates over a grid,
    represeting some sort of control over it.
    
    Its main method is update, which receives an action and a grid and returns
    a grid and a hidden state, both are used by the CAenv class to define an
    environment with the semantics for an Reinforcement Learning task.
    """
    # Set these in ALL subclasses
    # Data
    grid = None
    modifier_state = None
    
    # Services
    modifier = None
    automaton = None
    
    # Data Spaces
    grid_space = None
    modifier_state_space = None
    
    # RL Spaces
    observation_space = None # spaces.Tuple((grid_space, modifier_state_space))
    action_space = None
