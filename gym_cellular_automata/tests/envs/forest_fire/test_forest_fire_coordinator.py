from collections import Counter
from gym import spaces
import numpy as np

from gym_cellular_automata import Grid

from gym_cellular_automata.envs.forest_fire import ForestFireCellularAutomaton
from gym_cellular_automata.envs.forest_fire import ForestFireModifier
from gym_cellular_automata.envs.forest_fire import ForestFireCoordinator

from gym_cellular_automata.utils.config import get_forest_fire_config_dict
CONFIG = get_forest_fire_config_dict()

EMPTY = CONFIG['cell_symbols']['empty']
TREE  = CONFIG['cell_symbols']['tree']
FIRE  = CONFIG['cell_symbols']['fire']

CELL_STATES = CONFIG['cell_states']

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

EFFECTS = CONFIG['effects']

MAX_FREEZE = CONFIG['max_freeze']

CA_PARAMS_SPACE = ForestFireCellularAutomaton().context_space
GRID_SPACE = Grid(cell_states=CELL_STATES, shape=(ROW, COL)).grid_space
POS_SPACE = spaces.MultiDiscrete([ROW, COL])

ACTION_SPACE = spaces.Box(1, 9, shape=tuple(), dtype=np.uint8)

def instantiate_cellular_automaton():
    return ForestFireCellularAutomaton(grid_space    = GRID_SPACE,
                                       action_space  = ACTION_SPACE,
                                       context_space = CA_PARAMS_SPACE)

def instantiate_modifier():
    return ForestFireModifier(EFFECTS,
                              grid_space    = GRID_SPACE,
                              action_space  = ACTION_SPACE,
                              context_space = POS_SPACE)

def instantiate_coordinator():
    cellular_automaton = instantiate_cellular_automaton()
    modifier = instantiate_modifier()
    
    return ForestFireCoordinator(cellular_automaton,
                                 modifier,
                                 max_freeze = MAX_FREEZE)

def sample_coordinator_input():
    coordinator = instantiate_coordinator()
    
    grid = coordinator.grid_space.sample()
    action = coordinator.action_space.sample()
    context = coordinator.context_space.sample()
    
    return grid, action, context

def context_with_custom_freeze(custom = 1):
    ca_params = CA_PARAMS_SPACE.sample()
    pos       = POS_SPACE.sample()
    freeze = custom
    
    return ca_params, pos, freeze

# ------------ Tests

def test_API(
                operator = instantiate_coordinator()
            ):
    from gym_cellular_automata.tests import test_Operator_API_specifications
    test_Operator_API_specifications(operator)

def test_coordinator_output():
    coordinator = instantiate_coordinator()
    grid, action, context = sample_coordinator_input()
    
    new_grid, new_context = coordinator(grid, action, context)
    
    # Grid still does not completly behave as a numpy ndarray
    # new_grid[:] vs. just new_grid
    assert GRID_SPACE.contains(new_grid[:])
    assert coordinator.context_space.contains(new_context)

def test_coordinator_with_only_modifier():
    coordinator = instantiate_coordinator()
    
    grid, action, context = sample_coordinator_input()
    
    context = context_with_custom_freeze(1) 
    ca_params, pos, freeze = context
    
    new_grid, new_context = coordinator(grid, action, context)
    _, _, new_freeze  = new_context
    
    assert freeze - 1 == new_freeze

    which_cells_changed = np.not_equal(grid[:], new_grid[:]).flatten().tolist()
    how_many_changed = Counter(which_cells_changed)
    
    # At most 1 cell is changed    
    assert how_many_changed[True] <= 1

def test_coordinator_update_with_cellular_automaton_and_modifier():
    coordinator = instantiate_coordinator()
    max_freeze = coordinator.max_freeze
    grid, action, context = sample_coordinator_input()

    context = context_with_custom_freeze(0) 
    
    new_grid, new_context = coordinator(grid, action, context)    
    new_freeze = new_context[2]

    assert new_freeze == max_freeze
    # Testing the output grid is still needed it.
    # The update breaks the CA logic
    # specifically on the fire propagation rule
    # or does nothing
