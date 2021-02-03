import pytest
import numpy as np

from gym_cellular_automata import Grid

from gym_cellular_automata.envs.forest_fire import ForestFireModifier

from gym_cellular_automata.tests import test_Operator_API_specifications

EMPTY = 0
TREE = 1
FIRE = 2

CELL_STATES = 3

EFFECTS = {2: 0}

forest_fire_modifier = ForestFireModifier(EFFECTS)

test_Operator_API_specifications(forest_fire_modifier)

TEST_GRID = [[2,2,2],
             [2,2,2],
             [2,2,2]]

action_up_left = np.array(1)
action_up_center = np.array(2)
action_up_right = np.array(3)

action_middle_left = np.array(4)
action_middle_center = np.array(5)
action_middle_right = np.array(6)

action_down_left = np.array(7)
action_down_center = np.array(8)
action_down_right = np.array(9)

def test_ForestFireModifier_helicopter_movement():
    grid = Grid(TEST_GRID, cell_states=3)
    pos = np.array([1, 1])
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    row, col = forest_fire_modifier._move(grid, action_up_left, pos)
    assert row == 0 and col == 0
    row, col = forest_fire_modifier._move(grid, action_up_center, pos)
    assert row == 0 and col == 1
    row, col = forest_fire_modifier._move(grid, action_up_right, pos)
    assert row == 0 and col == 2
    
    row, col = forest_fire_modifier._move(grid, action_middle_left, pos)
    assert row == 1 and col == 0
    row, col = forest_fire_modifier._move(grid, action_middle_center, pos)
    assert row == 1 and col == 1
    row, col = forest_fire_modifier._move(grid, action_middle_right, pos)
    assert row == 1 and col == 2
    
    row, col = forest_fire_modifier._move(grid, action_down_left, pos)
    assert row == 2 and col == 0
    row, col = forest_fire_modifier._move(grid, action_down_center, pos)
    assert row == 2 and col == 1
    row, col = forest_fire_modifier._move(grid, action_down_right, pos)
    assert row == 2 and col == 2

def test_ForestFireModifier_helicopter_movement_boundaries():
    grid = Grid(TEST_GRID, cell_states=3)
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    corner_up_left = np.array([0, 0])
    corner_up_right = np.array([0, 2])
    corner_down_left = np.array([2, 0])
    corner_down_right = np.array([2, 2])
    
    row, col = forest_fire_modifier._move(grid, action_up_left, corner_up_left)
    assert row == 0 and col == 0
    row, col = forest_fire_modifier._move(grid, action_up_right, corner_up_right)
    assert row == 0 and col == 2
    
    row, col = forest_fire_modifier._move(grid, action_down_left, corner_down_left)
    assert row == 2 and col == 0 
    row, col = forest_fire_modifier._move(grid, action_down_right, corner_down_right)
    assert row == 2 and col == 2 

def test_ForestFireModifier_helicopter_illegal_actions():
    with pytest.raises(Exception):
        pos = np.array([1,1])
        forest_fire_modifier(TEST_GRID, 0, pos)
        forest_fire_modifier(TEST_GRID, -1, pos)
        forest_fire_modifier(TEST_GRID, 42, pos)
        forest_fire_modifier(TEST_GRID, 'foo', pos)
    
def test_ForestFireModifier_helicopter_fire_extinguish():
    grid = Grid(TEST_GRID, cell_states=3)
    pos = np.array([1, 1])
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    new_grid, (row, col) = forest_fire_modifier(grid, action_up_center, pos)
    assert new_grid[row, col] == EMPTY
    
    assert new_grid is grid, 'Same Object'
    
    new_grid, (row, col) = forest_fire_modifier(grid, action_down_center, pos)
    assert new_grid[row, col] == EMPTY
    
    assert new_grid is grid, 'Same Object'
