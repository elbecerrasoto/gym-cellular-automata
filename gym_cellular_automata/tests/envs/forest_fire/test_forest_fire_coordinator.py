from gym import spaces
import numpy as np

from gym_cellular_automata import Grid

from gym_cellular_automata.envs.forest_fire import ForestFireCellularAutomaton
from gym_cellular_automata.envs.forest_fire import ForestFireModifier
from gym_cellular_automata.envs.forest_fire import ForestFireCoordinator

CONFIG_FILE = 'gym_cellular_automata/envs/forest_fire/forest_fire_config.yaml'

def get_config_dict(file):
    import yaml
    yaml_file = open(file, 'r')
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content

CONFIG = get_config_dict(CONFIG_FILE)

EMPTY = CONFIG['cell_symbols']['empty']
TREE  = CONFIG['cell_symbols']['tree']
FIRE  = CONFIG['cell_symbols']['fire']

CELL_STATES = CONFIG['cell_states']

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

ACTIONS = CONFIG['actions']

EFFECTS = CONFIG['effects']

FREEZE_CA = CONFIG['freeze_ca']

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
                                 freeze_CA = FREEZE_CA)

def test_API(
                operator = instantiate_coordinator()
            ):
    from gym_cellular_automata.tests import test_Operator_API_specifications
    test_Operator_API_specifications(operator)

def sample_coordinator_input():
    coordinator = instantiate_coordinator()
    
    grid = coordinator.grid_space.sample()
    action = coordinator.action_space.sample()
    context = coordinator.context_space.sample()
    
    return grid, action, context

def test_coordinator_output():
    coordinator = instantiate_coordinator()
    grid, action, context = sample_coordinator_input()
    
    new_grid, new_context = coordinator(grid, action, context)
    
    # Grid still does not completly behave as a numpy ndarray
    # new_grid[:] vs. just new_grid
    assert GRID_SPACE.contains(new_grid[:])
    assert coordinator.context_space.contains(new_context)

# At 0 is updated       
# At 0 is replineshed

def test_coordinator_freeze_ca_decrease_logic():
    coordinator = instantiate_coordinator()
    max_freeze = coordinator.freeze_CA
    
    grid, action, context = sample_coordinator_input()
    
    freeze_ca = context[2]
    
    new_grid, new_context = coordinator(grid, action, context)
    
    new_freeze_ca = new_context[2]
    
    if freeze_ca != 0:
        assert new_freeze_ca == freeze_ca - 1
    else:
        assert new_freeze_ca == max_freeze      

def test_coordinator_freeze_ca_alternation_logic():
    pass

def test_coordinator_freeze_ca_does_not_depend_on_action():
    coordinator = instantiate_coordinator()
    grid, action, context = sample_coordinator_input()

    ca_params = CA_PARAMS_SPACE.sample()
    pos       = POS_SPACE.sample()
    freeze_ca = 1

    context = ca_params, pos, freeze_ca
    
    assert coordinator.context_space.contains(context)
    
    for key in ACTIONS:
        action = ACTIONS[key]

        new_grid, new_context = coordinator(grid, action, context)

        new_params, new_pos, new_freeze_ca = new_context

        assert new_freeze_ca == freeze_ca - 1
