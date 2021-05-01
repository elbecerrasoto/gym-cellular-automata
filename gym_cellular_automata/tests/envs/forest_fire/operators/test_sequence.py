from copy import copy

import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata.envs.forest_fire.operators import (
    Coordinate,
    ForestFire,
    Modify,
    Move,
    Sequence,
    WindyForestFire,
)
from gym_cellular_automata.grid_space import Grid
from gym_cellular_automata.operator import Identity

TESTS = 8
SEQUENCE = 8

EMPTY = 0
BURNED = 1
TREE = 3
FIRE = 25

ROW, COL = 8, 8


@pytest.fixture
def grid_space():
    return Grid(values=[EMPTY, BURNED, TREE, FIRE], shape=(ROW, COL))


@pytest.fixture
def operators(grid_space):

    dummy_space = spaces.Discrete(1)

    directions = {
        "up": {0},
        "down": {1},
        "left": {2},
        "right": {3},
        "not_move": {4},
    }

    effects = {EMPTY: TREE, FIRE: EMPTY, BURNED: TREE}

    max_freeze = 3

    identity = Identity(
        grid_space=grid_space, action_space=dummy_space, context_space=dummy_space
    )
    ca1 = ForestFire(
        EMPTY, BURNED, TREE, grid_space=grid_space, action_space=dummy_space
    )
    ca2 = WindyForestFire(
        EMPTY, BURNED, TREE, FIRE, grid_space=grid_space, action_space=dummy_space
    )
    move = Move(
        directions,
        grid_space=grid_space,
        action_space=spaces.Discrete(5),
        context_space=spaces.MultiDiscrete([ROW, COL]),
    )
    modify = Modify(
        effects,
        grid_space=grid_space,
        action_space=spaces.Discrete(2),
        context_space=spaces.MultiDiscrete([ROW, COL]),
    )
    coordinate = Coordinate(
        ca2,
        move,
        modify,
        max_freeze,
        grid_space=grid_space,
        action_space=spaces.Tuple(
            (ca2.action_space, move.action_space, modify.action_space, dummy_space)
        ),
        context_space=spaces.Tuple(
            (
                ca2.context_space,
                move.context_space,
                modify.context_space,
                spaces.Discrete(max_freeze + 1),
            )
        ),
    )

    return identity, ca1, ca2, move, modify, coordinate


@pytest.mark.repeat(TESTS)
def test_sequence(operators, grid_space):

    sequence = Sequence(operators)
    flow = np.random.choice(range(len(operators)), size=SEQUENCE, replace=True)

    grid = grid_space.sample()
    grid2 = grid.copy()

    actions = [operator.action_space.sample() for operator in operators]

    contexts = [operator.context_space.sample() for operator in operators]

    # Make the CAs deterministic
    contexts[1] = np.array([0.0, 0.0])
    contexts[2] = operators[2].context_space.high
    contexts[5] = list(contexts[5])
    contexts[5][0] = operators[5].suboperators.cellular_automaton.context_space.high

    contexts2 = copy(contexts)

    contexts_flow = contexts, flow

    obserseved_grid, __ = sequence(grid, actions, contexts_flow)

    def get_expected_grid(operators, igrid, actions, contexts, flow):

        for i in flow:

            ioperator = operators[i]
            iaction = actions[i]
            icontext = contexts[i]

            igrid, icontext = ioperator(igrid, iaction, icontext)
            contexts[i] = icontext

        return igrid

    expected_grid = get_expected_grid(operators, grid2, actions, contexts2, flow)

    assert np.all(obserseved_grid == expected_grid)
