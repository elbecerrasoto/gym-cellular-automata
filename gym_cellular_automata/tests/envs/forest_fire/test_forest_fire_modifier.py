import pytest
import numpy as np

from gym_cellular_automata import Grid
from gym_cellular_automata.envs.forest_fire import ForestFireModifier

TEST_GRID = [[2,2,2],
             [2,2,2],
             [2,2,2]]

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

EFFECTS = CONFIG['effects']

ACTION_UP_LEFT = CONFIG['actions']['up_left']
ACTION_UP_CENTER = CONFIG['actions']['up_center']
ACTION_UP_RIGHT = CONFIG['actions']['up_right']

ACTION_MIDDLE_LEFT = CONFIG['actions']['middle_left']
ACTION_MIDDLE_CENTER = CONFIG['actions']['middle_center']
ACTION_MIDDLE_RIGHT = CONFIG['actions']['middle_right']

ACTION_DOWN_LEFT = CONFIG['actions']['down_left']
ACTION_DOWN_CENTER = CONFIG['actions']['down_center']
ACTION_DOWN_RIGHT = CONFIG['actions']['down_right']

def test_API(
                operator = ForestFireModifier(EFFECTS)
            ):
    from gym_cellular_automata.tests import test_Operator_API_specifications
    test_Operator_API_specifications(operator)

def test_forest_fire_helicopter_movement():
    grid = Grid(TEST_GRID, cell_states=CELL_STATES)
    
    pos = np.array([1, 1])
    
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    # Up
    row, col = forest_fire_modifier._move(grid, ACTION_UP_LEFT, pos)
    assert row == 0 and col == 0
    row, col = forest_fire_modifier._move(grid, ACTION_UP_CENTER, pos)
    assert row == 0 and col == 1
    row, col = forest_fire_modifier._move(grid, ACTION_UP_RIGHT, pos)
    assert row == 0 and col == 2
    
    # Middle
    row, col = forest_fire_modifier._move(grid, ACTION_MIDDLE_LEFT, pos)
    assert row == 1 and col == 0
    row, col = forest_fire_modifier._move(grid, ACTION_MIDDLE_CENTER, pos)
    assert row == 1 and col == 1
    row, col = forest_fire_modifier._move(grid, ACTION_MIDDLE_RIGHT, pos)
    assert row == 1 and col == 2
    
    # Down
    row, col = forest_fire_modifier._move(grid, ACTION_DOWN_LEFT, pos)
    assert row == 2 and col == 0
    row, col = forest_fire_modifier._move(grid, ACTION_DOWN_CENTER, pos)
    assert row == 2 and col == 1
    row, col = forest_fire_modifier._move(grid, ACTION_DOWN_RIGHT, pos)
    assert row == 2 and col == 2

def test_ForestFireModifier_helicopter_movement_boundaries():
    
    grid = Grid(TEST_GRID, cell_states=3)
    
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    corner_up_left = np.array([0, 0])
    corner_up_right = np.array([0, 2])
    corner_down_left = np.array([2, 0])
    corner_down_right = np.array([2, 2])
    
    # Up Corners
    row, col = forest_fire_modifier._move(grid, ACTION_UP_LEFT, corner_up_left)
    assert row == 0 and col == 0
    row, col = forest_fire_modifier._move(grid, ACTION_UP_RIGHT, corner_up_right)
    assert row == 0 and col == 2
    
    # Down Corners
    row, col = forest_fire_modifier._move(grid, ACTION_DOWN_LEFT, corner_down_left)
    assert row == 2 and col == 0 
    row, col = forest_fire_modifier._move(grid, ACTION_DOWN_RIGHT, corner_down_right)
    assert row == 2 and col == 2 

def test_ForestFireModifier_helicopter_illegal_actions():
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    with pytest.raises(ValueError):
        pos = np.array([1,1])
        forest_fire_modifier(TEST_GRID, 0, pos)
        forest_fire_modifier(TEST_GRID, -1, pos)
        forest_fire_modifier(TEST_GRID, 42, pos)
        forest_fire_modifier(TEST_GRID, 'foo', pos)
    
def test_ForestFireModifier_helicopter_fire_extinguish():
    grid = Grid(TEST_GRID, cell_states=3)
    pos = np.array([1, 1])
    forest_fire_modifier = ForestFireModifier(EFFECTS)
    
    new_grid, (row, col) = forest_fire_modifier(grid, ACTION_UP_CENTER, pos)
    assert new_grid[row, col] == EMPTY
    
    assert new_grid is grid, 'Same Object'
    
    new_grid, (row, col) = forest_fire_modifier(grid, ACTION_DOWN_CENTER, pos)
    assert new_grid[row, col] == EMPTY
    
    assert new_grid is grid, 'Same Object'
