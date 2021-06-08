from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Optional

import numpy as np
from gym.spaces import Space
from objprint import add_objprint


@add_objprint
class Operator(ABC):

    # Set these in ALL subclasses
    suboperators: tuple = tuple()

    grid_dependant: Optional[bool] = None
    action_dependant: Optional[bool] = None
    context_dependant: Optional[bool] = None

    deterministic: Optional[bool] = None

    @abstractmethod
    def __init__(
        self,
        grid_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        context_space: Optional[Space] = None,
    ) -> None:

        # fmt: off
        self.grid_space    = grid_space    if grid_space    is not None else None
        self.action_space  = action_space  if action_space  is not None else None
        self.context_space = context_space if context_space is not None else None
        # fmt: on

    @abstractmethod
    def update(
        self, grid: np.ndarray, action: Any, context: Any
    ) -> tuple[np.ndarray, Any]:

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
    """The identity operator.
    It returns a hard copy grid and context.

    Shows the minimal implementation of an grid Operator.

    Useful for mocking grid Operators during testing.

        Example::

            >>> Identity()

    """

    grid_dependant = True
    action_dependant = False
    context_dependant = True

    deterministic = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, grid, action, context):
        return super().update(grid, action, context)
