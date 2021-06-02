from gym_cellular_automata import Operator


class Sequence(Operator):
    def __init__(self, operators, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = tuple(operators)

    def update(self, grid, actions, contexts_flow):
        contexts, flow = contexts_flow

        igrid = grid

        for i in flow:

            ioperator = self.suboperators[i]
            iaction = actions[i]
            icontext = contexts[i]

            igrid, icontext = ioperator(igrid, iaction, icontext)
            contexts[i] = icontext

        return igrid, (contexts, flow)


class SinglePass(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, operators, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = tuple(operators)

    def update(self, grid, subactions, subcontexts):
        subcontexts = list(subcontexts)

        for i, f in enumerate(self.suboperators):

            grid, icontext = f(grid, subactions[i], subcontexts[i])
            subcontexts[i] = icontext

        return grid, tuple(subcontexts)
