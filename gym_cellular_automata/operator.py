class Operator:

    # Set these in ALL subclasses
    suboperators = tuple()

    grid_space = None
    action_space = None
    context_space = None

    def update(self, grid, action, context):
        """
        Parameters
        ----------
        
        grid : array-like
            Cellular Automaton lattice.
        
        action : object
            Action influencing the operator output.
            Some operators do not use an action thus in that case
            this parameter would do nothing.
        
        context : object
            Extra information needed to compute the new_grid and new_context. 


        Returns
        -------
        
        new_grid, new_context : tuple
            
            new_grid : array-like
                Modified grid. 
            
            new_context : object
                Modified context.

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
