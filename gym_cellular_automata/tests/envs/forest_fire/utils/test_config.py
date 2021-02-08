import os
from gym_cellular_automata.envs.forest_fire.utils.config import get_forest_fire_config_dict
from gym_cellular_automata.envs.forest_fire.utils.config import FOREST_FIRE_CONFIG_FILE

def test_config_file_exists():
    assert os.path.isfile(FOREST_FIRE_CONFIG_FILE)

def test_config_filo_can_be_read_it_into_python():
    config = get_forest_fire_config_dict()
    assert isinstance(config, dict) 
