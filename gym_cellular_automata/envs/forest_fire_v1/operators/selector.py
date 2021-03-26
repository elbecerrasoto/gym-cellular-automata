# Perception and Induction

# The timing is clearer if it is on the environment.


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
