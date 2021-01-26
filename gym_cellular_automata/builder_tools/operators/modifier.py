from .operator import Operator

class Modifier(Operator):

    is_composition = False
    suboperators = tuple()
    
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    context_space = None
    
    effects = None

    def update(self, grid, action, context):
        raise NotImplementedError
