from gym_cellular_automata import Operator


class Modify(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(
        self,
        effects: dict,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.effects = effects

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):

        move_action, shoot_action = action

        row, col = context

        if shoot_action:

            grid[row, col] = self.effects[grid[row, col]]

        return grid, context
