class Operator:
    
    # Set these in ALL subclasses
    is_composition = None
    suboperators = tuple()
    
    grid_space = None
    action_space = None
    state_space = None
    
    def update(self, grid, action, state):
        raise NotImplementedError
