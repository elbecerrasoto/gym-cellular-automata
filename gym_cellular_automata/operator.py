from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Optional, Tuple

import numpy as np
from gymnasium.spaces import Space
from gymnasium.utils import seeding


class Operator(ABC):
    # Set these in ALL subclasses
    suboperators: Tuple = tuple()

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
        self.grid_space    = grid_space
        self.action_space  = action_space
        self.context_space = context_space
        # fmt: on

        self.seed()

    @abstractmethod
    def update(
        self, grid: np.ndarray, action: Any, context: Any
    ) -> Tuple[np.ndarray, Any]:
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
