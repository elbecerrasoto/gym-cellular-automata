import re
import numpy as np
from pathlib import Path

from gym import spaces

# Solves './gym_cellular_automata/envs/forest_fire_v1/'
forest_fire_dir = Path(__file__).parents[1]

FOREST_FIRE_CONFIG_FILE = forest_fire_dir / "forest_fire_v1.yaml"


def get_config_dict(file):
    import yaml

    yaml_file = open(file, "r")
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content


def get_forest_fire_config_dict():
    config = get_config_dict(FOREST_FIRE_CONFIG_FILE)

    config["effects"] = parse_effects(config)

    config["cell_type"] = parse_type(config["cell_type"])
    config["action_type"] = parse_type(config["action_type"])

    config["wind"] = parse_wind(config)

    return config


def parse_wind(config):
    WIND_TYPE = np.float64

    # fmt: off
    up_left    = config["wind_probs"]["up_left"]
    up         = config["wind_probs"]["up"]
    up_right   = config["wind_probs"]["up_right"]
    left       = config["wind_probs"]["left"]
    self       = config["wind_probs"]["self"]
    right      = config["wind_probs"]["right"]
    down_left  = config["wind_probs"]["down_left"]
    down       = config["wind_probs"]["down"]
    down_right = config["wind_probs"]["down_right"]
    
    wind = np.array(
        [[up_left,   up,     up_right],
         [left,      self,      right],
         [down_left, down, down_right]],
        dtype=WIND_TYPE
    )
    # fmt: on

    wind_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=WIND_TYPE)

    assert wind_space.contains(wind), "Bad Wind Data, check ranges [0.0-1.0]"

    return wind


def parse_effects(config):
    effects_str = config["effects"]
    effects = dict()

    for cell_name in effects_str:
        cell_symbol = config["cell_symbols"][cell_name]

        substitution_name = effects_str[cell_name]
        substitution = config["cell_symbols"][substitution_name]

        effects[cell_symbol] = substitution

    return effects


def parse_type(type_str):
    def remove_whitespace(my_str):
        return re.sub(r"\s+", "", my_str)

    type_str = remove_whitespace(type_str)

    re_uint8 = re.compile(r"uint8", re.IGNORECASE)
    re_uint16 = re.compile(r"uint16", re.IGNORECASE)
    re_uint32 = re.compile(r"uint32", re.IGNORECASE)
    re_uint64 = re.compile(r"uint64", re.IGNORECASE)
    re_uint = re.compile(r"uint", re.IGNORECASE)

    re_int8 = re.compile(r"int8", re.IGNORECASE)
    re_int16 = re.compile(r"int16", re.IGNORECASE)
    re_int32 = re.compile(r"int32", re.IGNORECASE)
    re_int64 = re.compile(r"int64", re.IGNORECASE)
    re_int = re.compile(r"int", re.IGNORECASE)

    # Unsigned int
    if re.search(re_uint8, type_str):
        return np.uint8

    elif re.search(re_uint16, type_str):
        return np.uint16

    elif re.search(re_uint32, type_str):
        return np.uint32

    elif re.search(re_uint64, type_str):
        return np.uint64

    elif re.search(re_uint, type_str):
        return np.uintc

    # int
    elif re.search(re_int8, type_str):
        return np.int8

    elif re.search(re_int16, type_str):
        return np.int16

    elif re.search(re_int32, type_str):
        return np.int32

    elif re.search(re_int64, type_str):
        return np.int64

    elif re.search(re_int, type_str):
        return np.intc

    else:
        return int


CONFIG = get_forest_fire_config_dict()
