from copy import copy

import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.operators import RepeatCA, WindyForestFire

TESTS = 8
SEQUENCE = 8

EMPTY = 0
BURNED = 1
TREE = 3
FIRE = 25

ROW, COL = 8, 8


@pytest.fixture
def grid_space():
    return GridSpace(values=[EMPTY, BURNED, TREE, FIRE], shape=(ROW, COL))


@pytest.fixture
def ca(grid_space):
    dummy_space = spaces.Discrete(1)
    return WindyForestFire(
        EMPTY, BURNED, TREE, FIRE, grid_space=grid_space, action_space=dummy_space
    )


@pytest.fixture
def repeat_ca(ca):
    def time_per_action(action):
        return 1.0

    def time_per_state(state):
        return 1.0

    return RepeatCA(ca, time_per_action, time_per_state)


# def test_repeat_ca_is_operator(repeat_ca):
#     from gym_cellular_automata.tests import assert_operator
#     assert_operator(repeat_ca)


def test_repeat_ca(ca, repeat_ca):

    grid = ca.grid_space.sample()
    grid02 = copy(grid)

    # Deterministic params
    params = ca.context_space.high
    accu = 0.0

    context = params, accu

    observed, new_context = repeat_ca(grid, None, context)

    __, observedc = new_context

    def get_expected(ca, grid, params):
        grid, __ = ca(grid, None, params)
        grid, __ = ca(grid, None, params)
        return grid

    expected = get_expected(ca, grid02, params)

    assert np.all(observed == expected)
    assert observedc == 0.0
