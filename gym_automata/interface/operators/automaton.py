from .operator import Operator

class Automaton(Operator):
    
    is_composition = False
    suboperators = tuple()
    
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action, state):
        raise NotImplementedError
