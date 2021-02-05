from gym import spaces

from gym_cellular_automata import Grid
from gym_cellular_automata.utils.neighbors import neighborhood_at
from gym_cellular_automata.envs.forest_fire import ForestFireCellularAutomaton

from gym_cellular_automata.utils.config import get_forest_fire_config_dict
CONFIG = get_forest_fire_config_dict()

# Steps to check CA rules
T_STEPS = 32
# Cells checked per step
TESTS_PER_STEP = 8

EMPTY = CONFIG['cell_symbols']['empty']
TREE  = CONFIG['cell_symbols']['tree']
FIRE  = CONFIG['cell_symbols']['fire']

CELL_STATES = CONFIG['cell_states']

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_col']

P_FIRE = CONFIG['ca_params']['p_fire']
P_TREE = CONFIG['ca_params']['p_tree']

def test_API(
                operator = ForestFireCellularAutomaton()
            ):
    from gym_cellular_automata.tests import test_Operator_API_specifications
    test_Operator_API_specifications(operator)

def test_forest_fire_cell_symbols():
    ca_operator = ForestFireCellularAutomaton()
    
    assert ca_operator.empty == EMPTY
    assert ca_operator.tree == TREE
    assert ca_operator.fire == FIRE
    
def test_forest_fire_update_on_tree_ring():
    ca_operator = ForestFireCellularAutomaton()
    
    TREE_RING = Grid([[1,1,1],
                      [1,2,1],
                      [1,1,1]], cell_states=CELL_STATES)
    grid = TREE_RING
    
    ca_params = P_FIRE, P_TREE
    
    grid, _ = ca_operator(grid, None, ca_params)
    
    assert grid[1,1] == EMPTY, '1st Ring Update'
    assert grid[0,0] == FIRE,  '1st Ring Update'
    assert grid[0,2] == FIRE,  '1st Ring Update'
    assert grid[1,0] == FIRE,  '1st Ring Update'
    assert grid[1,2] == FIRE,  '1st Ring Update'
    
    grid, _ = ca_operator(grid, None, ca_params)
    
    assert grid[0,0] == EMPTY, '2nd Ring Update'
    assert grid[0,2] == EMPTY, '2nd Ring Update'
    assert grid[1,0] == EMPTY, '2nd Ring Update'
    assert grid[1,2] == EMPTY, '2nd Ring Update'
    
def assert_forest_fire_update_at_position_row_col(grid, new_grid, row, col):
    log_error = f'\n row: {row}' + \
                f'\n col: {col}' + \
                f'\n\n grid: {grid}' + \
                f'\n\n new_grid: {new_grid}'
    
    old_cell_value = grid[row, col]
    new_cell_value = new_grid[row, col]
    neighborhood = neighborhood_at(grid, (row, col), invariant=EMPTY)
    
    # Explicit rules
    if old_cell_value == TREE and FIRE in neighborhood:
        # A TREE next to a FIRE turns into a FIRE.
        assert new_cell_value == FIRE, 'NON Fire Propagation' + log_error
    
    if old_cell_value == FIRE:
        # A FIRE turns into an EMPTY.
        assert new_cell_value == EMPTY, 'NON Fire Consumption' + log_error
    
    # Implicit rules
    if old_cell_value == EMPTY:
        # An EMPTY never turns into FIRE.
        assert new_cell_value != FIRE, 'Empty Combustion' + log_error
    
    if old_cell_value == TREE:
        # A TREE never turns into EMPTY.
        assert new_cell_value != EMPTY, 'Dying Trees' + log_error
    
    if old_cell_value == FIRE:
        # A FIRE never turns into a TREE.
        assert new_cell_value != TREE, 'Created by Fire' + log_error
        # A FIRE never turns into a FIRE.
        assert new_cell_value != FIRE, 'Lingering Fire' + log_error

def test_forest_fire_update():
    ca_operator = ForestFireCellularAutomaton()
    
    grid = Grid(shape=(ROW, COL), cell_states=CELL_STATES)    
    pos_space = spaces.MultiDiscrete([ROW, COL])
    
    for step in range(T_STEPS):
        ca_params = ca_operator.context_space.sample()
        
        new_grid, _ = ca_operator(grid, None, ca_params)
        
        assert grid is not new_grid, 'ca_operator is returning the same grid object'
        
        for test in range(TESTS_PER_STEP):
            row, col = pos_space.sample()

            assert_forest_fire_update_at_position_row_col(grid, new_grid, row, col)
            
        grid = new_grid
