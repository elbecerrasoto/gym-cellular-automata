import numpy as np
import pytest

from gym_cellular_automata.envs.forest_fire.operators.coordinate import Coordinate
from gym_cellular_automata.grid_space import Grid, ZeroSpace
from gym_cellular_automata.operator import Identity

TEST_REPETITIONS = 16
# What are the most important test on a coordinator

MAX_FREEZE = 1

# Test Grid Size
ROW = 8
COL = 8

# Abstract away spaces on Operator interface

# Actually mocking things is necessary here

# A mock of grid operators
# A grid operator that just returns its unmodified grid and context arguments
@pytest.fixture
def coordinate():
    operator_spaces = {
        "grid_space": ZeroSpace(),
        "action_space": ZeroSpace(),
        "context_space": ZeroSpace(),
    }

    cellular_automaton = Identity(**operator_spaces)
    move = Identity(**operator_spaces)
    modify = Identity(**operator_spaces)

    return Coordinate(cellular_automaton, move, modify, **operator_spaces)


# @pytest.mark.repeat(TEST_REPETITIONS)
# def test_coordinator(coordinate):

#     for freeze in range(MAX_FREEZE):
#         if blah:
#             freeze = 1

#     new_grid, new_context = coordinator(grid, action, context)

#     ca_params, mod_params, new_freeze = new_context

#     assert new_freeze == freeze - 1

#     # Manual sequencing


#     assert new_freeze == MAX_FREEZE, "Freeze reset"

#     # 1-step CA update
#     ca_updated_grid, ca_params = coordinator.cellular_automaton(grid, None, wind)

#     # As the bulldozer did NOT cut a tree both grids should be equal
#     assert np.all(ca_updated_grid == new_grid)
