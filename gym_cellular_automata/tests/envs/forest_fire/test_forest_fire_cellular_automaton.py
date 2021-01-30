import numpy as np
from gym import spaces

from gym_cellular_automata import Grid
from gym_cellular_automata.utils.neighbors import neighborhood_at
from gym_cellular_automata.envs.forest_fire.forest_fire_cellular_automaton import ForestFireCellularAutomaton

from gym_cellular_automata.tests.operators.test_operator import test_Operator_API_specifications
from gym_cellular_automata.tests.operators.test_cellular_automaton import test_CellularAutomaton_API_specifications

EMPTY = 0
TREE = 1
FIRE = 2

CELL_STATES = 3

T_STEPS = 64
TESTS_PER_STEP = 8

ROW = 5
COL = 5

grid = Grid(shape = (ROW, COL), cell_states = CELL_STATES)
forest_fire_CA = ForestFireCellularAutomaton(grid.grid_space)

test_Operator_API_specifications(forest_fire_CA)
test_CellularAutomaton_API_specifications(forest_fire_CA)

def test_ForestFireCellularAutomaton_cell_symbols_specifications(forest_fire_CA = forest_fire_CA):
    assert forest_fire_CA.empty == EMPTY
    assert forest_fire_CA.tree == TREE
    assert forest_fire_CA.fire == FIRE
    
TREE_CIRCLE = np.array([[1,1,1],
                        [1,2,1],
                        [1,1,1]])

tree_circle = Grid(data=TREE_CIRCLE, cell_states=3)

def test_CellularAutomaton_grid_update_tree_circle(grid = tree_circle):
    assert grid[1,1] == FIRE, '0th Ring Update'
    
    forest_fire_CA = ForestFireCellularAutomaton(grid.grid_space)
    
    context = forest_fire_CA.context_space.sample()
    grid, _ = forest_fire_CA(grid, None, context)
    
    assert grid[1,1] == EMPTY, '1st Ring Update'
    assert grid[0,0] == FIRE, '1st Ring Update'
    assert grid[0,2] == FIRE, '1st Ring Update'
    assert grid[1,0] == FIRE, '1st Ring Update'
    assert grid[1,2] == FIRE, '1st Ring Update'
    
    context = forest_fire_CA.context_space.sample()
    grid, _ = forest_fire_CA(grid, None, context)
    
    assert grid[0,0] == EMPTY, '2nd Ring Update'
    assert grid[0,2] == EMPTY, '2nd Ring Update'
    assert grid[1,0] == EMPTY, '2nd Ring Update'
    assert grid[1,2] == EMPTY, '1nd Ring Update'
    
def assert_forest_fire_update_at(grid, new_grid, row, col):
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

def test_CellularAutomaton_grid_update_general():
    grid = Grid(shape=(ROW, COL), cell_states=CELL_STATES)
    
    pos_space = spaces.MultiDiscrete([ROW, COL])
    forest_fire_CA = ForestFireCellularAutomaton(grid.grid_space)
    
    for step in range(T_STEPS):
        
        forest_fire_params = forest_fire_CA.context_space.sample()
        
        new_grid, _ = forest_fire_CA.update(grid,
                                     action = None,
                                     context = forest_fire_params)
        
        assert not grid is new_grid, 'Same object'
        
        for test in range(TESTS_PER_STEP):
            pos = pos_space.sample()
            row, col = pos

            assert_forest_fire_update_at(grid, new_grid, row, col)
            
        grid = new_grid
