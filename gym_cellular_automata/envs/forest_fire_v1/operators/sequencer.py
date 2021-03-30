from math import modf

from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


# ------------ Forest Fire Operator Sequencer


class Sequencer(Operator):

    # fmt: off
    _t_none   = CONFIG["time"]["none_action"]
    _t_move   = CONFIG["time"]["move_action"]
    _t_shoot  = CONFIG["time"]["shoot_action"]
  
    _movement = CONFIG["actions"]["movement"]
    _shooting = CONFIG["actions"]["shooting"]
    # fmt: on

    def __init__(
        self, operators, grid_space=None, action_space=None, context_space=None
    ):

        self.suboperators = tuple(operators)

        self._init_temporal_mappings()

    def update(self, grid, action, context):

        op_contexts, temporal_params = context

        # Operation sequence per step only depends on actions and temporal params
        operation_flow, new_temporal_params = self._get_operation_flow(
            grid, action, temporal_params
        )

        # Apply Operator Sequence over grid
        # Assume the same action could be applied to each Operator
        for operation in operation_flow:

            operator = self.operators[operation]
            op_context = op_contexts[operation]

            new_grid, new_context = operator(grid, action, op_context)

            grid = new_grid
            op_contexts[operation] = new_context

        new_context = op_contexts, temporal_params

        return new_grid, new_context

    def _get_operation_flow(self, grid, action, context):

        movement, shooting = action

        accumulated_time = context

        # Mapping of actions ---> to time (on units of CA updates)
        t_move = self.movement_timings[movement]
        t_shoot = self.shooting_timings[shooting]

        t_taken = t_move + t_shoot

        accumulated_time += t_taken

        # Decimal and Integer parts
        accumulated_time, ca_computations = modf(accumulated_time)

        # Operartors order is: CA, Modifier
        operation_flow = int(ca_computations) * [0] + [1]

        return operation_flow, accumulated_time

    def _init_action_time_mappings(self):

        self.movement_timings = {move: self._t_move for move in self._movement.values()}
        self.shooting_timings = {
            shoot: self._t_shoot for shoot in self._shooting.values()
        }

        self.movement_timings[self._shooting["not_move"]] = self._t_none
        self.shooting_timings[self._shooting["none"]] = self._t_none
