from gym_cellular_automata._config import PROJECT_PATH
from gym_cellular_automata.forest_fire.utils.config import (
    get_config_dict,
    group_actions,
    parse_wind,
    translate,
)

CONFIG_PATH = (
    PROJECT_PATH / "./gym_cellular_automata/forest_fire/bulldozer/bulldozer.yaml"
)


def get_forest_fire_config_dict():
    config = get_config_dict(CONFIG_PATH)

    config["effects"] = translate(config["effects"], config["cell_symbols"])

    config["actions"]["sets"] = group_actions(config["actions"]["movement"])

    config["wind"] = parse_wind(config["wind_probs"])

    return config


CONFIG = get_forest_fire_config_dict()
