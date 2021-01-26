from .operator import Operator
from .cellular_automaton import CellularAutomaton
from .modifier import Modifier

class Coordinator(Operator):

    is_composition = True
    
    # Set these in ALL subclasses
    suboperators = (CellularAutomaton(), Modifier())
    
    cellular_automaton = suboperators[0]
    modifier = suboperators[1]
    
    grid_space = None
    action_space = None
    context_space = None

    def update(self, grid, action, context):
        raise NotImplementedError
