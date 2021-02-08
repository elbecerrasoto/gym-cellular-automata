from functools import partial
from pathlib import Path

FOREST_FIRE_CONFIG_FILE = 'gym_cellular_automata/envs/forest_fire/forest_fire_config.yaml'
FOREST_FIRE_CONFIG_FILE = Path(FOREST_FIRE_CONFIG_FILE)

def get_config_dict(file):
    import yaml
    yaml_file = open(file, 'r')
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content

get_forest_fire_config_dict = partial(get_config_dict,
                                      file = FOREST_FIRE_CONFIG_FILE)
