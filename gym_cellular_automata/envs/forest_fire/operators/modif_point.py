from abc import ABC
from collections import namedtuple

from gym_cellular_automata import Operator
from gym_cellular_automata.operator import Identity


class Modify(ABC, Operator):
    def __init__(
        self,
        move=None,
        shoot=None,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        if move is None or shoot is None:
            self.suboperators = suboperators(move=Identity(), shoot=Identity())

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):

        # Change position based on action
        grid, context.position = self.suboperators.move(grid, action, context.position)

        # Apply grid point modifications
        grid, context = self.suboperators.shoot(grid, action, context)

        return grid, context
