# Load the YAML configuration file into a python dict


def get_config_dict(file):
    import yaml

    yaml_file = open(file)
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content


# Post YAML loading dynamic modifications
# Common parsing operations on Forest Fire CA Environments


def parse_effects(config, kw_effects="effects", kw_cell_symbols="cell_symbols"):
    effects_str = config["effects"]
    effects = dict()

    for cell_name in effects_str:
        cell_symbol = config["cell_symbols"][cell_name]

        substitution_name = effects_str[cell_name]
        substitution = config["cell_symbols"][substitution_name]

        effects[cell_symbol] = substitution

    return effects


def parse_actions(config):
    up_left = config["actions"]["movement"]["up_left"]
    up = config["actions"]["movement"]["up"]
    up_right = config["actions"]["movement"]["up_right"]

    left = config["actions"]["movement"]["left"]
    not_move = config["actions"]["movement"]["not_move"]
    right = config["actions"]["movement"]["right"]

    down_left = config["actions"]["movement"]["down_left"]
    down = config["actions"]["movement"]["down"]
    down_right = config["actions"]["movement"]["down_right"]

    up_set = {up_left, up, up_right}
    down_set = {down_left, down, down_right}

    left_set = {up_left, left, down_left}
    right_set = {up_right, right, down_right}

    not_move_set = {not_move}

    return {
        "up": up_set,
        "down": down_set,
        "left": left_set,
        "right": right_set,
        "not_move": not_move_set,
    }
