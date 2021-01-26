from .operator import Operator

class CellularAutomaton(Operator):
    
    is_composition = False
    suboperators = tuple()
    
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    context_space = None

    def update(self, grid, action, context):
        raise NotImplementedError
