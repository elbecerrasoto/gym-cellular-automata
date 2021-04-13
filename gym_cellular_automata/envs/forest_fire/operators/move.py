from collections.abc import Hashable

import numpy as np
from gym import logger

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    are_my_neighbors_a_boundary,
)


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

        row, col = context

        is_boundary = are_my_neighbors_a_boundary(grid, (row, col))
        # Change semantics for clarity
        valid_up, valid_down, valid_left, valid_right = [
            not boundary for boundary in is_boundary
        ]

        if valid_up and move_action in self.up_set:
            row -= 1

        if valid_down and move_action in self.down_set:
            row += 1

        if valid_left and move_action in self.left_set:
            col -= 1

        if valid_right and move_action in self.right_set:
            col += 1

        return grid, np.array([row, col])
