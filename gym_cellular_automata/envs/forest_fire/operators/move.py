import numpy as np

from gym_cellular_automata import Operator


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
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.effects = effects

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):

        move_action, shoot_action = action

        row, col = context

        is_boundary = are_my_neighbors_a_boundary(grid, (row, col))

        new_row = (
            row - 1
            if not is_boundary.up and int(move_action) in self.up_set
            else row + 1
            if not is_boundary.down and int(move_action) in self.down_set
            else row
        )

        new_col = (
            col - 1
            if not is_boundary.left and int(move_action) in self.left_set
            else col + 1
            if not is_boundary.right and int(move_action) in self.right_set
            else col
        )

        return grid, np.array([new_row, new_col])
