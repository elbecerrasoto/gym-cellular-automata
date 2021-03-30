from math import modf
import numpy as np
from gym_cellular_automata import Operator
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


class Sequencer(Operator):

    # fmt: off
    _t_none  = 1e-4
    _t_move  = 0.024
    _t_shoot = 0.120
  
    _up_set    = CONFIG["actions"]["sets"]["up"]
    _down_set  = CONFIG["actions"]["sets"]["down"]
    
    _left_set  = CONFIG["actions"]["sets"]["left"]
    _right_set = CONFIG["actions"]["sets"]["right"]
    
    _shoot = CONFIG["actions"]["shooting"]["shoot"]
    _none  = CONFIG["actions"]["shooting"]["none"]

    # fmt: on

    def __init__(
        self, operators, grid_space=None, action_space=None, context_space=None
    ):

        self.suboperators = tuple(operators)

    def update(self, grid, action, context):

        op_contexts, temporal_params = context

        operation_flow, new_temporal_params = self._get_operation_flow(
            grid, action, context
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

        op_contexts, temporal_params = context

        # Mappings of actions taken ---> to time on units of CA updates
        t_move = self.movement_timings[movement]
        t_shoot = self.shooting_timings[shooting]

        t_taken = t_move + t_shoot

        self.accumulated_time += t_taken
        # Decimal and Integer parts
        self.accumulated_time, ca_computations = modf(self.accumulated_time)
        # Assuming the following order:
        # operators = CA, Bulldozer
        operation_flow_idxs = int(ca_computations) * [0] + [1]
        return operation_flow_idxs
