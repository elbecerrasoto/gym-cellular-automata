import math
from typing import Callable

import numpy as np

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.operator import Operator


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
        self.deterministic = self.ca.deterministic

    def update(self, grid, action, context):
        ca_params, accu_time = context

        time_action = self.t_acting(action)
        time_state = self.t_perception((grid, context))
        time_taken = time_action + time_state

        accu_time += time_taken
        accu_time, repeats = math.modf(accu_time)

        for repeat in range(int(repeats)):
            grid, ca_params = self.ca(grid, action, ca_params)

        return grid, (ca_params, np.array(accu_time, dtype=TYPE_BOX))
