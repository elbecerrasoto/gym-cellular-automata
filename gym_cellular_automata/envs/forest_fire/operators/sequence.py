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
