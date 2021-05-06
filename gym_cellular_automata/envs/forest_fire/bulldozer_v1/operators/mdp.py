from gym_cellular_automata import Operator


class MDP(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, repeat_ca, move, modify, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = repeat_ca, move, modify

        self.repeat_ca = repeat_ca
        self.move = move
        self.modify = modify

    def update(self, grid, action, context):

        amove, ashoot = action
        ca_params, time, position = context

        grid, (ca_params, time) = self.repeat_ca(grid, action, (ca_params, time))

        grid, position = self.move(grid, amove, position)
        grid, position = self.modify(grid, ashoot, position)

        return grid, (ca_params, time, position)
