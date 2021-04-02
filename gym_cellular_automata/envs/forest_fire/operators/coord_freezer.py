import numpy as np

from gym import spaces
from gym_cellular_automata import Operator

# ------------ Forest Fire Coordinator


class Freezer(Operator):
    def __init__(
        self,
        cellular_automaton,
        modifier,
        max_freeze,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.suboperators = cellular_automaton, modifier
        self.cellular_automaton, self.modifier = cellular_automaton, modifier

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):
        ca_params, mod_params, freeze = context

        freeze = int(freeze)

        if freeze == 0:

            grid, ca_params = self.cellular_automaton(grid, action, ca_params)

            grid, mod_params = self.modifier(grid, action, mod_params)

            freeze = np.array(self.max_freeze)

        else:

            grid, mod_params = self.modifier(grid, action, mod_params)

            freeze = np.array(freeze - 1)

        context = ca_params, mod_params, freeze

        return grid, context
