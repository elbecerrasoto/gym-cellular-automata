import os

import numpy as np
import pytest

from gym_cellular_automata.envs.forest_fire.v0.utils.config import (
    FOREST_FIRE_CONFIG_FILE,
    get_forest_fire_config_dict,
    parse_type,
)


def test_config_file_exists():
    assert os.path.isfile(FOREST_FIRE_CONFIG_FILE)


def test_config_filo_can_be_read_it_into_python():
    config = get_forest_fire_config_dict()
    assert isinstance(config, dict)


@pytest.mark.parametrize(
    "type_str, expected",
    [
        ("integer", np.intc),
        ("UINT8", np.uint8),
        ("Int16", np.int16),
        ("uint32", np.uint32),
        (" i n t  \t  3 2", np.int32),
        ("::uint64", np.uint64),
        ("Hello World!", int),
    ],
)
def test_parse_type(type_str, expected):
    inferred = parse_type(type_str)
    print(f"input is {type_str}")
    print(f"inferred is {inferred}")
    print(f"expected is {expected}")
    assert inferred is expected


def test_parse_type_error():
    with pytest.raises(TypeError):
        parse_type(-1)
