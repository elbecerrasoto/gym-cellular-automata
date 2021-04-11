from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import copy


class Operator(ABC, Callable):

    # Set these in ALL subclasses
    suboperators = tuple()

    grid_dependant = None
    action_dependant = None
    context_dependant = None

    grid_space = None
    action_space = None
    context_space = None

    @abstractmethod
    def update(self, grid, action, context):

        """Update a Cellular Automaton's Lattice (Grid) by using a provided action and context.

        Parameters
        ----------

        grid : array-like
            Cellular Automaton lattice.

        action : object
            Action influencing the operator output.

        context : object
            Extra information.


        Returns
        -------
        new_grid : array-like
            Modified grid.

        new_context : object
            Modified context.

        """

        new_grid = copy(grid)
        new_context = copy(context)

        return new_grid, new_context

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class Identity(Operator):

    grid_dependant = True
    action_dependant = False
    context_dependant = True

    def update(self, grid, action, context):
        return super.update(grid, action, context)
