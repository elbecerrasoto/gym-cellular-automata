import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    are_my_neighbors_a_boundary,
)
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


# fmt: off
UP_LEFT    = CONFIG["actions"]["movement"]["up_left"]
UP         = CONFIG["actions"]["movement"]["up"]
UP_RIGHT   = CONFIG["actions"]["movement"]["up_right"]

LEFT       = CONFIG["actions"]["movement"]["left"]
NOT_MOVE   = CONFIG["actions"]["movement"]["not_move"]
RIGHT      = CONFIG["actions"]["movement"]["right"]

DOWN_LEFT  = CONFIG["actions"]["movement"]["down_left"]
DOWN       = CONFIG["actions"]["movement"]["down"]
DOWN_RIGHT = CONFIG["actions"]["movement"]["down_right"]

SHOOT = CONFIG["actions"]["shooting"]["shoot"]
NONE  = CONFIG["actions"]["shooting"]["none"]

UP_SET    = {UP_LEFT,   UP,    UP_RIGHT}
DOWN_SET  = {DOWN_LEFT, DOWN,  DOWN_RIGHT}

LEFT_SET  = {UP_LEFT,   LEFT,  DOWN_LEFT}
RIGHT_SET = {UP_RIGHT,  RIGHT, DOWN_RIGHT}
# fmt: on


# ------------ Forest Fire Modifier


class Bulldozer(Operator):
    is_composition = False

    def __init__(self, effects, grid_space=None, action_space=None, context_space=None):

        self.effects = effects

        if action_space is None:
            action_space = spaces.MultiDiscrete(
                [len(CONFIG["actions"]["movement"]), len(CONFIG["actions"]["shooting"])]
            )

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

        if shooting == SHOOT:
            for symbol in self.effects:
                if grid[row, col] == symbol:
                    grid[row, col] = self.effects[symbol]

        return grid, new_position

    def _move(self, grid, action, pos):
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
