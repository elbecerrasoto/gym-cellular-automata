from collections import namedtuple

import numpy as np
from gym import spaces

from gym_cellular_automata import Operator


class Coordinate(Operator):

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move", "modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, cellular_automaton, move, modify, max_freeze, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = self.Suboperators(cellular_automaton, move, modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):

        ca_action, move_action, modify_action, coordinate_action = action
        ca_context, move_context, modify_context, coordinate_context = context

        freeze = int(coordinate_context)
        # assert np.all(move_context == modify_context) # Gets in the way of sampling from context_space
        position = move_context

        def move_then_modify(grid, move_action, modify_action, position):

            grid, position = self.suboperators.move(grid, move_action, position)
            grid, position = self.suboperators.modify(grid, modify_action, position)

            return grid, position

        if freeze == 0:

            grid, ca_params = self.suboperators.cellular_automaton(
                grid, ca_action, ca_context
            )

            grid, position = move_then_modify(
                grid, move_action, modify_action, position
            )

            freeze = np.array(self.max_freeze)

        else:

            grid, position = move_then_modify(
                grid, move_action, modify_action, position
            )

            freeze = np.array(freeze - 1)

        context = ca_context, position, position, freeze

        return grid, context
