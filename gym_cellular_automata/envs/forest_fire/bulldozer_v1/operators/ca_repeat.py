import math
from typing import Callable

import numpy as np

from gym_cellular_automata import Operator


class SinglePass(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, operators, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = tuple(operators)

    def update(self, grid, subactions, subcontexts):
        subcontexts = list(subcontexts)

        for i, f in enumerate(self.suboperators):

            grid, icontext = f(grid, subactions[i], subcontexts[i])
            subcontexts[i] = icontext

        return grid, tuple(subcontexts)


class RepeatCA(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(
        self,
        cellular_automaton,
        t_acting: Callable,
        t_perception: Callable,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.t_acting = t_acting
        self.t_perception = t_perception

        self.ca = cellular_automaton
        self.suboperators = (self.ca,)

    def update(self, grid, action, context):
        ca_params, accu_time = context

        time_action = self.t_acting(action)
        time_state = self.t_perception((grid, context))
        time_taken = time_action + time_state

        accu_time += time_taken
        accu_time, repeats = math.modf(accu_time)

        for repeat in range(int(repeats)):
            grid, ca_params = grid, icontext = self.ca(grid, action, ca_params)

        return grid, (ca_params, np.array(accu_time))
