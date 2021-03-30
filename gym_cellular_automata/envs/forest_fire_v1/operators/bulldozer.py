import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    are_my_neighbors_a_boundary,
)
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


# ------------ Forest Fire Modifier


class Bulldozer(Operator):

    # fmt:off
    _effects = CONFIG["effects"]
   
    _up_set    = CONFIG["actions"]["sets"]["up"]
    _down_set  = CONFIG["actions"]["sets"]["down"]
    
    _left_set  = CONFIG["actions"]["sets"]["left"]
    _right_set = CONFIG["actions"]["sets"]["right"]
    
    _shoot = CONFIG["actions"]["shooting"]["shoot"]
    _none  = CONFIG["actions"]["shooting"]["none"]
    
    _n_moves         = len(CONFIG["actions"]["movement"])
    _n_shoots        = len(CONFIG["actions"]["shooting"])
    # fmt: on

    def __init__(self, grid_space=None, action_space=None, context_space=None):

        if action_space is None:
            action_space = spaces.MultiDiscrete([self._n_moves, self._n_shoots])

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):

        if not self.action_space.contains(action):
            raise ValueError(f"action: {action} does not belong to {self.action_space}")

        position = context

        movement, shooting = action

        new_position = self._move(grid, movement, position)

        row, col = new_position

        if shooting == self._shoot:

            for symbol in self._effects:

                if grid[row, col] == symbol:

                    grid[row, col] = self._effects[symbol]

        return grid, new_position

    def _move(self, grid, action, pos):
        row, col = pos

        is_boundary = are_my_neighbors_a_boundary(grid, pos)

        new_row = (
            row - 1
            if not is_boundary.up and int(action) in self._up_set
            else row + 1
            if not is_boundary.down and int(action) in self._down_set
            else row
        )

        new_col = (
            col - 1
            if not is_boundary.left and int(action) in self._left_set
            else col + 1
            if not is_boundary.right and int(action) in self._right_set
            else col
        )

        return np.array([new_row, new_col])
