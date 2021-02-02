from gym import spaces
import numpy as np

from gym_cellular_automata import Grid

from gym_cellular_automata.tests.operators import test_Coordinator_API_specifications

from gym_cellular_automata.envs.forest_fire import ForestFireCellularAutomaton
from gym_cellular_automata.envs.forest_fire import ForestFireModifier
from gym_cellular_automata.envs.forest_fire import ForestFireCoordinator

EMPTY = 0
TREE = 1
FIRE = 2

ROW = 3
COL = 3

# The coordinator makes the assertions
# the other needs do not necessarily need the shit

# Coordinator take the spaces from the inferior operations
# Or uses the supplied values

# The coordinator must have the asserts
# Delay making the decision

grid = Grid(cell_states=3, shape=(ROW, COL))
EFFECTS = {FIRE: EMPTY}

GRID_SPACE = spaces.Box(0, 2, shape=(ROW, COL), dtype=np.uint16)
POS_SPACE = spaces.MultiDiscrete([ROW, COL])
ACTION_SPACE = spaces.Box(1, 9, shape=tuple(), dtype=np.uint8)

forest_fire_cellular_automaton = \
    ForestFireCellularAutomaton(grid_space = GRID_SPACE,
                                action_space = ACTION_SPACE)
forest_fire_modifier = \
    ForestFireModifier(EFFECTS,
                       grid_space = GRID_SPACE,
                       action_space = ACTION_SPACE,
                       context_space = POS_SPACE)

forest_fire_coordinator= ForestFireCoordinator(forest_fire_cellular_automaton,
                                                forest_fire_modifier,
                                                freeze_CA=2,
                      grid_space=None, action_space=None, context_space=None)

test_Coordinator_API_specifications(forest_fire_coordinator)

# To test
# action does nothing
# Returns grid, return pos, freeze
# At 0 is updated
# At 0 is replineshed

grid = GRID_SPACE.sample()
action = ACTION_SPACE.sample()
context = forest_fire_coordinator.context_space.sample()

list(enumerate(grid))


probs, pos, steps_until_CA = context

x, y = probs

forest_fire_coordinator(grid, action, context)

def test_ForestFireCoordinator_update_output():
    forest_fire_coordinator()

def test_ForestFireCoordinator_action_does_nothing():
    pass



def test_ForestFireCoordinator_steps_until_CA_logic():
    pass


