from gym_cellular_automata.envs.forest_fire.utils.config import (
    get_config_dict,
    get_path,
    group_actions,
    parse_wind,
    translate,
)

CONFIG_FILE = "bulldozer_v1.yaml"


def get_forest_fire_config_dict():
    config = get_config_dict(get_path(CONFIG_FILE, __file__, behind=1))

    config["effects"] = translate(config["effects"], config["cell_symbols"])

    config["actions"]["sets"] = group_actions(config["actions"]["movement"])

    config["wind"] = parse_wind(config["wind_probs"])

    return config


CONFIG = get_forest_fire_config_dict()
