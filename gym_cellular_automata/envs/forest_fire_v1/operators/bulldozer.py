import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    are_my_neighbors_a_boundary,
)
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


# ------------ Forest Fire Modifier


class Bulldozer(Operator):
    is_composition = False

    # fmt:off
    _effects = CONFIG["effects"]
    
    _up_left    = CONFIG["actions"]["movement"]["up_left"]
    _up         = CONFIG["actions"]["movement"]["up"]
    _up_right   = CONFIG["actions"]["movement"]["up_right"]
    
    _left       = CONFIG["actions"]["movement"]["left"]
    _not_move   = CONFIG["actions"]["movement"]["not_move"]
    _right      = CONFIG["actions"]["movement"]["right"]
    
    _down_left  = CONFIG["actions"]["movement"]["down_left"]
    _down       = CONFIG["actions"]["movement"]["down"]
    _down_right = CONFIG["actions"]["movement"]["down_right"]
    
    _up_set    = {_up_left,   _up,    _up_right}
    _down_set  = {_down_left, _down,  _down_right}
    
    _left_set  = {_up_left,   _left,  _down_left}
    _right_set = {_up_right,  _right, _down_right} 
    
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
