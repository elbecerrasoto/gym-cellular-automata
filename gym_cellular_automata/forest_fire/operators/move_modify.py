import numpy as np
from gym import logger, spaces

from gym_cellular_automata.operator import Operator


class Move(Operator):

    grid_dependant = (
        False  # If a constant size grid is used (that is usually the case).
    )
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, directions_sets, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # fmt: off
        self.up_set       = directions_sets["up"]
        self.down_set     = directions_sets["down"]
        self.left_set     = directions_sets["left"]
        self.right_set    = directions_sets["right"]
        self.not_move_set = directions_sets["not_move"]
        # fmt: on

        self.movement_set = (
            self.up_set
            | self.down_set
            | self.left_set
            | self.right_set
            | self.not_move_set
        )

    def update(self, grid, action, context):

        # A common input is a scalar of type ndarray
        action = int(action)

        def get_new_position(position: tuple) -> np.array:
            row, col = position

            nrows, ncols = grid.shape

            # fmt: off
            valid_up    = row > 0
            valid_down  = row < (nrows - 1)
            valid_left  = col > 0
            valid_right = col < (ncols - 1)

            if (action in self.up_set)    and valid_up:
                row -= 1

            if (action in self.down_set)  and valid_down:
                row += 1

            if (action in self.left_set)  and valid_left:
                col -= 1

            if (action in self.right_set) and valid_right:
                col += 1
            # fmt: on

            return np.array([row, col])

        return grid, get_new_position(context)


class Modify(Operator):
    hit = False

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, effects: dict, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.effects = effects

    def update(self, grid, action, context):
        self.hit = False

        row, col = context

        if action:

            if grid[row, col] in self.effects:

                grid[row, col] = self.effects[grid[row, col]]
                self.hit = True

        return grid, context


class MoveModify(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, move, modify, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.suboperators = move, modify

        self.move = move
        self.modify = modify

        if self.action_space is None:
            if (
                self.move.action_space is not None
                and self.move.action_space is not None
            ):
                self.action_space = spaces.Tuple(
                    (self.move.action_space, self.move.action_space)
                )

        if self.context_space is None:
            if (
                self.move.context_space is not None
                and self.modify.context_space is not None
            ):
                assert self.move.context_space == self.modify.context_space
                self.context_space = self.move.context_space

    def update(self, grid, subactions, position):
        move_action, modify_action = subactions

        grid, position = self.move(grid, move_action, position)
        grid, position = self.modify(grid, modify_action, position)

        return grid, position
