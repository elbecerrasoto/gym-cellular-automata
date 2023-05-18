from copy import copy

import numpy as np
import pytest
from gymnasium import spaces

from gym_cellular_automata.forest_fire.operators import RepeatCA, WindyForestFire
from gym_cellular_automata.grid_space import GridSpace

TESTS = 8
SEQUENCE = 8

EMPTY = 0
TREE = 3
FIRE = 25

ROW, COL = 8, 8


@pytest.fixture
def grid_space():
    return GridSpace(values=[EMPTY, TREE, FIRE], shape=(ROW, COL))


@pytest.fixture
def dummy_space():
    return spaces.Discrete(1)


@pytest.fixture
def ca(grid_space, dummy_space):
    return WindyForestFire(
        EMPTY, TREE, FIRE, grid_space=grid_space, action_space=dummy_space
    )


@pytest.fixture
def context_space(ca):
    accu_time_space = spaces.Box(np.array(0.0), np.array(1.0))
    return spaces.Tuple((ca.context_space, accu_time_space))


@pytest.fixture
def repeat_ca(ca, grid_space, dummy_space, context_space):
    def time_per_action(action):
        return 1.0

    def time_per_state(state):
        return 1.0

    return RepeatCA(
        ca,
        time_per_action,
        time_per_state,
        grid_space=grid_space,
        action_space=dummy_space,
        context_space=context_space,
    )


def test_repeat_ca_is_operator(repeat_ca):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(repeat_ca, strict=True)


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
