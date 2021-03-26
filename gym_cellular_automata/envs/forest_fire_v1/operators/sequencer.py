"""
Induction based timings.

As opposed to Perception based.

"""
from math import modf

# fmt: off

# Small t_none guarantees termination
T_NONE  = 1e-4
T_MOVE  = 0.04
T_SHOOT = 0.16

# fmt: on

action = 4, 5

movement, shooting = action

t_shoot = shooting_timings[shooting]
t_move = movement_timings[shooting]

t_taken = t_move + t_move

accumulated_time += t_taken

accumulated_time, ca_computations = modf(accumulated_time)

# Do shit
int(accumulated_time)


ca_op

operators = ()

# CA o B
# CA^n o B


# Temporal Sequencing
def get_operation_flow(operators, state, action):
    return ...


grid = ...
operators = ...
contexts = ...
action = ...


operation_flow = get_operation_flow(operators, action, state=(grid, contexts))

for i_op in operation_flow:

    operator = operators[i_op]
    context = contexts[i_op]

    new_grid, new_context = operator(grid, action, context)

    grid = new_grid
    contexts[i_op] = new_context
