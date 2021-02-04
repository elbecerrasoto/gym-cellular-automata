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
            assert cellular_automaton.grid_space is not None, 'grid_space could not be inferred' 
            self.grid_space = cellular_automaton.grid_space
        
        if action_space is None:
            assert modifier.action_space is not None, 'action_space could not be inferred'
            self.action_space = modifier.action_space
            
        if context_space is None:
            assert cellular_automaton.context_space is not None, 'context_space could not be inferred'
            assert modifier.context_space is not None, 'context_space could not be inferred'
            CA_params_space = cellular_automaton.context_space
            pos_space = modifier.context_space
            self.context_space = spaces.Tuple((CA_params_space,
                                               pos_space,
                                               self.freeze_CA_space))

    def update(self, grid, action, context):
        CA_params, pos, steps_until_CA = context
        steps_until_CA = int(steps_until_CA)
 
        if steps_until_CA == 0:

            grid, _ = self.cellular_automaton(grid, action, CA_params)
            grid, new_pos = self.modifier(grid, action, pos)
            
            steps_until_CA = self.freeze_CA
            
        else:
            
            grid, new_pos = self.modifier(grid, action, pos)
            
            steps_until_CA -= 1
        
        context = CA_params, new_pos, steps_until_CA
        return grid, context
