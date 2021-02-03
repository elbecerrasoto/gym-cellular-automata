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

action_up_left = np.array(1)
action_up_center = np.array(2)
action_up_right = np.array(3)

action_middle_left = np.array(4)
action_middle_center = np.array(5)
action_middle_right = np.array(6)

action_down_left = np.array(7)
action_down_center = np.array(8)
action_down_right = np.array(9)

ACTIONS = (action_up_left, action_up_center, action_up_right,
           action_middle_left, action_middle_center, action_middle_right,
           action_down_left, action_down_center, action_down_right)


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
ACTION_SPACE = spaces.Box(1, 9, shape=tuple())


forest_fire_cellular_automaton = \
    ForestFireCellularAutomaton(grid_space = GRID_SPACE,
                                action_space = ACTION_SPACE)
forest_fire_modifier = \
    ForestFireModifier(EFFECTS,
                       grid_space = GRID_SPACE,
                       action_space = ACTION_SPACE,
                       context_space = POS_SPACE)

forest_fire_coordinator = ForestFireCoordinator(forest_fire_cellular_automaton,
                                                forest_fire_modifier,
                                                freeze_CA=2,
                      grid_space=None, action_space=None, context_space=None)

test_Coordinator_API_specifications(forest_fire_coordinator)


FREEZE_CA_SPACE = forest_fire_coordinator.freeze_CA_space


# To test
# action does nothing
# Returns grid, return pos, freeze
# At 0 is updated
# At 0 is replineshed

PARAMS_SPACE = forest_fire_cellular_automaton.context_space
PARAMS_SPACE


def test_ForestFireCoordinator_update_output():
    grid = GRID_SPACE.sample()
    action = ACTION_SPACE.sample()
    context = forest_fire_coordinator.context_space.sample()
    
    new_grid, new_context = forest_fire_coordinator(grid, action, context)
    
    # still does not behaves as a numpy array
    assert GRID_SPACE.contains(new_grid[:])
    assert forest_fire_coordinator.context_space.contains(new_context)

def test_ForestFireCoordinator_until_ca_does_not_depends_on_action():
    
    grid = GRID_SPACE.sample()
    params = PARAMS_SPACE.sample()
    pos = POS_SPACE.sample()
    # freeze_ca = np.array(1) ERROR why?
    freeze_ca = 1
    
    # context = params, pos, np.array(freeze_ca) ERROR why?
    context = params, pos, freeze_ca

    for action in ACTIONS:
        print(f'freeze_ca: {freeze_ca}')
        new_grid, new_context = forest_fire_coordinator(grid, action, context)
        new_params, new_pos, new_freeze_ca = new_context
        print(f'freeze_ca: {freeze_ca}')
        assert new_freeze_ca == freeze_ca - 1

def test_ForestFireCoordinator_steps_until_CA_logic():
    pass
