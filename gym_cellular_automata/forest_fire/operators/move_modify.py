from collections.abc import Hashable

import numpy as np
from gym import logger

from gym_cellular_automata import Operator


def hashable(x):
    return isinstance(x, Hashable)


class Move(Operator):

    grid_dependant = True  # A minor effect because of boundaries
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

        if not hashable(action):
            casting = int
            logger.warn(f"Unhashable Movement Action {action}.\nCasting to {casting}.")
            action = casting(action)

        if action not in self.movement_set:
            logger.warn(
                f"Movement Action {action} not in set {self.movement_set}.\nPosition will not change."
            )

        def get_new_position(position: tuple) -> np.array:
            row, col = position

            n_row, n_col = grid.shape

            # fmt: off
            valid_up    = row > 0
            valid_down  = row < (n_row - 1)
            valid_left  = col > 0
            valid_right = col < (n_col - 1)

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

    def update(self, grid, subactions, position):
        move_action, modify_action = subactions

        grid, position = self.move(grid, move_action, position)
        grid, position = self.modify(grid, modify_action, position)

        return grid, position
