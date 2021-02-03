from gym import spaces
from gym_cellular_automata import Operator

class ForestFireCoordinator(Operator):
    
    def __init__(self, cellular_automaton, modifier, freeze_CA,
                 grid_space=None, action_space=None, context_space=None):
        
        self.suboperators = cellular_automaton, modifier
        self.cellular_automaton, self.modifier = cellular_automaton, modifier
        
        self.freeze_CA = freeze_CA
        self.freeze_CA_space = spaces.Discrete(freeze_CA + 1)  
        
        if grid_space is None:
            self.grid_space = cellular_automaton.grid_space
        
        if action_space is None:
            self.action_space = modifier.action_space
            
        if context_space is None:
            CA_params_space = self.cellular_automaton.context_space
            pos_space = self.modifier.context_space
            self.context_space = spaces.Tuple((CA_params_space,
                                               pos_space,
                                               self.freeze_CA_space))

    def update(self, grid, action, context):
        CA_params, pos, steps_until_CA = context
 
        if steps_until_CA == 0:

            grid, _ = self.cellular_automaton(grid, action, CA_params)
            grid, new_pos = self.modifier(grid, action, pos)
            
            steps_until_CA = self.freeze_CA
            
        else:
            
            grid, new_pos = self.modifier(grid, action, pos)
            
            steps_until_CA -= 1
        
        context = CA_params, new_pos, steps_until_CA
        return grid, context
