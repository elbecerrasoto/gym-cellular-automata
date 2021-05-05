import math
from operator import mul
from typing import Callable

from collection import namedtuple

from gym_cellular_automata import Operator


class RepeatCA(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    SubOperators = namedtuple("SubOperators", ["sequence"])

    def __init__(
        self,
        sequence,
        tacting: Callable,
        tperception: Callable,
        get_subactions: Callable = mul,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.suboperators = self.SubOperators(sequence)

    def update(self, grid, action, context):
        nops = len(self.suboperators.sequence)

        subcontexts, accu_time = context

        # Repeat the same action
        subactions = self.get_subactions(action)

        time_action = self.tacting(action)
        time_state = self.tperception((grid, context))

        time_taken = time_action + time_state

        accu_time += time_taken
        accu_time, car = math.modf(accu_time)

        # Operartors order is: Apply CA 'car' times, then the others only once
        opsflow = int(car) * [0] + [i for i in range(1, nops)]

        subcontexts_opsflow = subcontexts, opsflow

        grid, subcontexts = self.sequence(grid, subactions, subcontexts_opsflow)

        return grid, (subcontexts, accu_time)
