from .operator import Operator

class Exchanger(Operator):

    is_composition = False
    suboperators = tuple()
    
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    state_space = None
    
    effects = None

    def update(self, grid, action, state):
        raise NotImplementedError
