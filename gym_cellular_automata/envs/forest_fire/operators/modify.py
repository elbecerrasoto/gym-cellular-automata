from gym_cellular_automata import Operator


class Modify(Operator):
    hit = False

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, effects: dict, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.effects = effects

    def update(self, grid, action, context):
        self.hit = False

        row, col = context

        if action:

            if grid[row, col] in self.effects:

                grid[row, col] = self.effects[grid[row, col]]
                self.hit = True

        return grid, context
