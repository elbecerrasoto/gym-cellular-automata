import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    are_my_neighbors_a_boundary,
)

from ..utils.config import CONFIG

ACTION_UP_LEFT = CONFIG["actions"]["up_left"]
ACTION_UP = CONFIG["actions"]["up"]
ACTION_UP_RIGHT = CONFIG["actions"]["up_right"]

ACTION_LEFT = CONFIG["actions"]["left"]
ACTION_NOT_MOVE = CONFIG["actions"]["not_move"]
ACTION_RIGHT = CONFIG["actions"]["right"]

ACTION_DOWN_LEFT = CONFIG["actions"]["down_left"]
ACTION_DOWN = CONFIG["actions"]["down"]
ACTION_DOWN_RIGHT = CONFIG["actions"]["down_right"]

UP_SET = {ACTION_UP_LEFT, ACTION_UP, ACTION_UP_RIGHT}
DOWN_SET = {ACTION_DOWN_LEFT, ACTION_DOWN, ACTION_DOWN_RIGHT}

LEFT_SET = {ACTION_UP_LEFT, ACTION_LEFT, ACTION_DOWN_LEFT}
RIGHT_SET = {ACTION_UP_RIGHT, ACTION_RIGHT, ACTION_DOWN_RIGHT}

ACTION_TYPE = CONFIG["action_type"]

# ------------ Forest Fire Modifier


class ForestFireModifier(Operator):
    hit = False

    def __init__(self, effects, grid_space=None, action_space=None, context_space=None):

        self.effects = effects

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        new_pos = self._move(grid, action, pos=context)
        row, col = new_pos

        self.hit = False

        # If applicable, cell state changes from FIRE to EMPTY.
        for symbol in self.effects:

            if grid[row, col] == symbol:
                grid[row, col] = self.effects[symbol]
                self.hit = True

        return grid, new_pos

    def _move(self, grid, action, pos):
        action = np.array(action)

        if not self.action_space.contains(action):
            raise ValueError(f"action: {action} does not belong to {self.action_space}")

        row, col = pos

        is_boundary = are_my_neighbors_a_boundary(grid, pos)

        new_row = (
            row - 1
            if not is_boundary.up and int(action) in UP_SET
            else row + 1
            if not is_boundary.down and int(action) in DOWN_SET
            else row
        )

        new_col = (
            col - 1
            if not is_boundary.left and int(action) in LEFT_SET
            else col + 1
            if not is_boundary.right and int(action) in RIGHT_SET
            else col
        )

        return np.array([new_row, new_col])
