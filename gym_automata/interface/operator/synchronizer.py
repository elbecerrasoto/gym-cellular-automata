from .operator import Operator

class Synchronizer(Operator):

    is_composition = True
    suboperators = tuple()
    
    # Set these in ALL subclasses
    automaton = None
    exchanger = None
    
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action):
        raise NotImplementedError
