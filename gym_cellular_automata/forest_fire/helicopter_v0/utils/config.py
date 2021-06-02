from gym_cellular_automata.forest_fire.utils.config import (
    get_config_dict,
    get_path,
    group_actions,
    translate,
)

CONFIG_NAME = "helicopter_v0.yaml"


def get_forest_fire_config_dict():
    config = get_config_dict(get_path(CONFIG_NAME, __file__))

    config["effects"] = translate(config["effects"], config["cell_symbols"])

    config["actions_sets"] = group_actions(config["actions"])

    return config


CONFIG = get_forest_fire_config_dict()
