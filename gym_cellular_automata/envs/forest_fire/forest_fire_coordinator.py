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
