import pytest

from gym_cellular_automata.envs.forest_fire.operators.coordinate import Coordinate
from gym_cellular_automata.operator import Identity

MAX_FREEZE = 1


@pytest.fixture
def coordinate():

    cellular_automaton = Identity()
    move = Identity()
    modify = Identity()

    return Coordinate(cellular_automaton, move, modify, MAX_FREEZE)


def test_coordinator(coordinate):

    for freeze in range(MAX_FREEZE):

        grid = None
        action = None
        context = None, None, freeze

        __, observed_context = coordinate(grid, action, context)
        __, __, observed_freeze = observed_context

    if freeze != 0:

        expected_freeze = freeze - 1
        assert observed_freeze == expected_freeze

    else:

        expected_freeze = MAX_FREEZE
        assert observed_freeze == expected_freeze
