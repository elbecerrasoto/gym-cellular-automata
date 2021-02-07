import pytest
import numpy as np

from gym_cellular_automata.envs.forest_fire.utils.neighbors import neighborhood_at
from gym_cellular_automata.envs.forest_fire.utils.neighbors import are_my_neighbors_a_boundary

from gym_cellular_automata.envs.forest_fire.utils.config import get_forest_fire_config_dict

CONFIG = get_forest_fire_config_dict()

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

EMPTY = CONFIG['cell_symbols']['empty']
TREE  = CONFIG['cell_symbols']['tree']
FIRE  = CONFIG['cell_symbols']['fire']

CELL_STATES = CONFIG['cell_states']
CELL_TYPE = np.uint8

@pytest.fixture
def singleton_2d():
    return np.array([[2]], dtype=CELL_TYPE)

@pytest.fixture
def fire_x_grid():
    return np.array([[2, 0, 2],
                     [0, 2, 1],
                     [2, 1, 2]], dtype=CELL_TYPE)

def test_boundaries_of_singleton(singleton_2d):
    my_boundaries = are_my_neighbors_a_boundary(grid = singleton_2d,
                                            pos = (0,0))

    assert my_boundaries.up    is True
    assert my_boundaries.down  is True
    assert my_boundaries.left  is True
    assert my_boundaries.right is True
    
def test_neighbors_of_singleton(singleton_2d):
    neighbors = neighborhood_at(grid = singleton_2d,
                                pos = (0,0),
                                invariant = EMPTY)
    
    assert neighbors.up_left       == EMPTY
    assert neighbors.up_center     == EMPTY
    assert neighbors.up_right      == EMPTY
    
    assert neighbors.middle_left   == EMPTY
    assert neighbors.middle_center == FIRE
    assert neighbors.middle_right  == EMPTY
    
    assert neighbors.down_left     == EMPTY
    assert neighbors.down_center   == EMPTY
    assert neighbors.down_right    == EMPTY

def test_boundaries_of_fire_x_grid(fire_x_grid):
    my_boundaries = are_my_neighbors_a_boundary(grid = fire_x_grid,
                                                pos = (1,1))
    
    assert my_boundaries.up    is False
    assert my_boundaries.down  is False
    assert my_boundaries.left  is False
    assert my_boundaries.right is False
    
    my_boundaries = are_my_neighbors_a_boundary(grid = fire_x_grid,
                                                pos = (2,0))
    
    assert my_boundaries.up    is False
    assert my_boundaries.down  is True
    assert my_boundaries.left  is True
    assert my_boundaries.right is False

def test_neighbors_of_fire_x_grid(fire_x_grid):
    neighbors = neighborhood_at(grid = fire_x_grid,
                                pos = (2,0),
                                invariant = EMPTY)
    
    assert neighbors.up_left       == EMPTY
    assert neighbors.up_center     == EMPTY
    assert neighbors.up_right      == FIRE
    
    assert neighbors.middle_left   == EMPTY
    assert neighbors.middle_center == FIRE
    assert neighbors.middle_right  == TREE
    
    assert neighbors.down_left     == EMPTY
    assert neighbors.down_center   == EMPTY
    assert neighbors.down_right    == EMPTY
