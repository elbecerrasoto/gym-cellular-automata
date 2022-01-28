from gym_cellular_automata.operator import Operator


class Identity(Operator):
    """The identity operator.
    It returns a hard copy grid and context.

    Shows the minimal implementation of an grid Operator.

    Useful for mocking grid Operators during testing.

        Example::

            >>> Identity()

    """

    grid_dependant = True
    action_dependant = False
    context_dependant = True

    deterministic = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, grid, action, context):
        return super().update(grid, action, context)
