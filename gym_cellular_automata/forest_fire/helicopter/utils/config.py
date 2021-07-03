from gym_cellular_automata import PROJECT_PATH
from gym_cellular_automata.forest_fire.utils.config import (
    get_config_dict,
    group_actions,
    translate,
)

CONFIG_PATH = (
    PROJECT_PATH / "./gym_cellular_automata/forest_fire/helicopter/helicopter_v0.yaml"
)


def get_forest_fire_config_dict():
    config = get_config_dict(CONFIG_PATH)

    config["effects"] = translate(config["effects"], config["cell_symbols"])

    config["actions_sets"] = group_actions(config["actions"])

    return config


CONFIG = get_forest_fire_config_dict()
