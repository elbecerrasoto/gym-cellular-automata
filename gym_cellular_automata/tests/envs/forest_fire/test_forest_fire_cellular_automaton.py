import numpy as np
from gym import spaces

from gym_cellular_automata.builder_tools.data import Grid
from gym_cellular_automata.utils.neighbors import neighborhood_at
from gym_cellular_automata.envs.forest_fire.forest_fire_operators import ForestFireCellularAutomaton

from gym_cellular_automata.tests.builder_tools.operators.test_operator import test_Operator_API_specifications
from gym_cellular_automata.tests.builder_tools.operators.test_cellular_automaton import test_CellularAutomaton_API_specifications

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
    assert grid[1,1] == FIRE
    
    forest_fire_CA = ForestFireCellularAutomaton(grid.grid_space)
    
    context = forest_fire_CA.context_space.sample()
    new_grid, _ = forest_fire_CA(grid, None, context)
    
    assert grid[1,1] == EMPTY
    assert grid[0,0] == FIRE
    assert grid[0,2] == FIRE
    assert grid[1,0] == FIRE
    assert grid[1,2] == FIRE
    
    context = forest_fire_CA.context_space.sample()
    new_grid, _ = forest_fire_CA(grid, None, context)
    
    assert grid[0,0] == EMPTY
    assert grid[0,2] == EMPTY
    assert grid[1,0] == EMPTY
    assert grid[1,2] == EMPTY
    
def assert_forest_fire_update_at(grid, pos, new_cell_value):

    old_cell_value = grid[pos]
    neighborhood = neighborhood_at(grid, pos, invariant=EMPTY)
    
    # Explicit rules
    # A TREE next to a FIRE turns into a FIRE.
    fire_propagation = True
    if old_cell_value == TREE and FIRE in neighborhood:
       fire_propagation = new_cell_value == FIRE
    
    # A FIRE turns into an EMPTY.
    fire_consumption = True
    if old_cell_value == FIRE:
        fire_consumption = new_cell_value == EMPTY
    
    # Implicit rules
    # An EMPTY never turns into FIRE.
    empty_combustion = False
    if old_cell_value == EMPTY:
        empty_combustion = new_cell_value == FIRE
    
    # A TREE never turns into EMPTY.
    dying_trees = False
    if old_cell_value == TREE:
        dying_trees = new_cell_value == EMPTY
    
    # A FIRE never turns into a TREE.
    creation_by_fire = False
    # A FIRE never turns into a FIRE.
    lingering_fire = False
    if old_cell_value == FIRE:
        creation_by_fire = new_cell_value == TREE
        lingering_fire = new_cell_value == FIRE
        
    positive_controls = fire_consumption and fire_propagation
    negative_controls = not(empty_combustion or dying_trees or creation_by_fire or lingering_fire)
    
    return positive_controls and negative_controls

def test_CellularAutomaton_grid_update_general():
    grid = Grid(shape=(ROW, COL), cell_states=CELL_STATES)
    pos_space = spaces.MultiDiscrete([ROW, COL])
    forest_fire_CA = ForestFireCellularAutomaton(grid.grid_space)
    
    for step in range(T_STEPS):
        new_grid, _ = forest_fire_CA(grid,
                                     None,
                                     forest_fire_CA.context_space.sample())
        
        for test in range(TESTS_PER_STEP):
            pos = pos_space.sample()
            row, col = pos
            assert assert_forest_fire_update_at(grid, pos, new_grid[row][col])
            
        grid = new_grid
