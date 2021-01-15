from .operator import Operator
from .automaton import Automaton
from .exchanger import Exchanger

class Synchronizer(Operator):

    is_composition = True
    
    # Set these in ALL subclasses
    suboperators = (Automaton(), Exchanger())
    
    automaton = suboperators[0]
    exchanger = suboperators[1]
    
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action, state):
        raise NotImplementedError
