import numpy as np

from gym import spaces
from gym_cellular_automata import Operator

# ------------ Forest Fire Coordinator


null = spaces.Discrete(1)

null.sample()
null.contains('a')


null = spaces.Box(np.array(0), np.array(0))

null = spaces.Box(np.array(0), np.array(0))

null.sample()
null.contains(np.array(0.0001))

# The Null space captures all my exception behavior.


class Coordinator(Operator):
    is_composition = True

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

        if grid_space is None:
            assert (
                cellular_automaton.grid_space is not None
            ), "grid_space could not be inferred"

            self.grid_space = cellular_automaton.grid_space

        if action_space is None:
            assert (
                modifier.action_space is not None
            ), "action_space could not be inferred"

            self.action_space = modifier.action_space

        if context_space is None:
            assert (
                cellular_automaton.context_space is not None
            ), "context_space could not be inferred"
            assert (
                modifier.context_space is not None
            ), "context_space could not be inferred"

            self.ca_params_space = cellular_automaton.context_space
            self.pos_space = modifier.context_space

            # It is flattened
            self.context_space = spaces.Tuple(
                (self.ca_params_space, self.pos_space, self.freeze_space)
            )

    def update(self, grid, action, context):
        ca_params, mod_params, freeze = context
        
        # Permisive on what you accept
        freeze = int(freeze)

        if freeze == 0:

            grid, ca_params = self.cellular_automaton(grid, action, mod_params)

            grid, mod_params = self.modifier(grid, action, mod_params)

            freeze = np.array(self.max_freeze)

        else:

            grid, mod_params = self.modifier(grid, action, mod_params)

            # Strict on what you return
            freeze = np.array(freeze - 1)

        context = ca_params, mod_params, freeze

        return grid, context
