from collections.abc import Hashable

import numpy as np
from gym import logger

from gym_cellular_automata import Operator


def hashable(x):
    return isinstance(x, Hashable)


class Move(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(
        self,
        up_set: set,
        down_set: set,
        left_set: set,
        right_set: set,
        not_move_set: set,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

        self.up_set = up_set
        self.down_set = down_set
        self.left_set = left_set
        self.right_set = right_set
        self.not_move_set = not_move_set

        self.movement_set = up_set | down_set | left_set | right_set | not_move_set

    def update(self, grid, action, context):

        move_action, shoot_action = action

        if not hashable(move_action):
            casting = int
            logger.warn(
                f"Unhashable Movement Action {move_action}.\nCasting to {casting}."
            )
            move_action = casting(move_action)

        if move_action not in self.movement_set:
            logger.warn(
                f"Movement Action {move_action} not in set {self.movement_set}.\nPosition will not change."
            )

        def get_new_position(position: tuple) -> np.array:
            row, col = position

            n_row, n_col = grid.shape

            valid_up = row > 0
            valid_down = row < (n_row - 1)
            valid_left = col > 0
            valid_right = col < (n_col - 1)

            # fmt: off
            if (move_action in self.up_set)    and valid_up:
                row -= 1

            if (move_action in self.down_set)  and valid_down:
                row += 1

            if (move_action in self.left_set)  and valid_left:
                col -= 1

            if (move_action in self.right_set) and valid_right:
                col += 1
            # fmt: on

            return np.array([row, col])

        return grid, get_new_position(context)
