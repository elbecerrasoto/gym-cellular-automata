import numpy as np


# Load the YAML configuration file into a python dict
def get_config_dict(file):
    import yaml

    yaml_file = open(file)
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content


###############################################################################
# Utilities
# Common parsing operations on Forest Fire CA Environments
# Post YAML loading, dynamic modifications
###############################################################################


# Useful for changing Effects on string to Effects on int representation
def translate(mapping: dict, translation: dict) -> dict:
    m = mapping
    f = translation
    return {f[x]: f[m[x]] for x in m}


# Group movement actions, useful to instantiate a "move operator"
def group_actions(actions: dict) -> dict:
    import re

    def group_byRegex(actions: dict, regex: str) -> set:
        return {actions[action] for action in actions if re.search(regex, action)}

    REGEXES = (r"^up", r"^down", r"left$", r"right$", r"not_move")

    up, down, left, right, not_move = map(
        lambda regex: group_byRegex(actions, regex), REGEXES
    )

    return {
        "up": up,
        "down": down,
        "left": left,
        "right": right,
        "not_move": not_move,
    }


def parse_wind(windD: dict) -> np.ndarray:
    from gym import spaces

    # fmt: off
    wind = np.array(
        [
            [ windD["up_left"]  , windD["up"]  , windD["up_right"]   ],
            [ windD["left"]     ,    0.0       , windD["right"]      ],
            [ windD["down_left"], windD["down"], windD["down_right"] ],
        ]
    )
    # fmt: on
    wind_space = spaces.Box(0.0, 1.0, shape=(3, 3))

    assert wind_space.contains(wind), "Bad Wind Data, check ranges [0.0-1.0]"

    return wind
