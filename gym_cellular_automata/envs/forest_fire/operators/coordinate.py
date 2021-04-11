from collections import namedtuple

import numpy as np
from gym import spaces

from gym_cellular_automata import Operator

# ------------ Forest Fire Coordinator


class Coordinate(Operator):

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move", "modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(
        self,
        cellular_automaton,
        move,
        modify,
        max_freeze,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.suboperators = self.Suboperators(cellular_automaton, move, modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):

        ca_params, position, freeze = context

        freeze = int(freeze)

        if freeze == 0:

            grid, ca_params = self.suboperators.cellular_automaton(
                grid, action, ca_params
            )
            grid, position = self.suboperators.move(grid, action, position)
            grid, position = self.suboperators.modify(grid, action, position)

            freeze = np.array(self.max_freeze)

        else:

            grid, position = self.suboperators.move(grid, action, ca_params)
            grid, mod_params = self.suboperators.modify(grid, action, ca_params)

            freeze = np.array(freeze - 1)

        context = ca_params, position, freeze

        return grid, context
