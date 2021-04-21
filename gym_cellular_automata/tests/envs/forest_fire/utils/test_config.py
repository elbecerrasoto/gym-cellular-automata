import pytest

from gym_cellular_automata.envs.forest_fire.utils.config import get_config_dict

# Test temporal file
YAML = "test.yaml"

# Test values for YAML parsing
# fmt: off
TEST_PRIMITIVES = {
    "INT"    : { "input": "0"            , "expected": 0     },
    "INT2"   : { "input": "-1"           , "expected": -1    },
    "FLOAT"  : { "input": "0.99"         , "expected": 0.99  },
    "FLOAT2" : { "input": "!!float 1e-4" , "expected": 1e-4  },
    "STR"    : { "input": "foo"          , "expected": "foo" },
    "STR2"   : { "input": "'1'"          , "expected": "1"   },
}
# fmt: on


@pytest.fixture
def content():
    content_str = ""

    for primitive in TEST_PRIMITIVES:
        line = primitive + ": " + TEST_PRIMITIVES[primitive]["input"] + "\n"
        content_str += line

    return content_str


def test_int_float_str_parsing(tmp_path, content):
    path = tmp_path / YAML
    path.write_text(content)

    observed_content = get_config_dict(path)

    assert isinstance(observed_content, dict)

    for primitive in TEST_PRIMITIVES:

        observed = observed_content[primitive]
        expected = TEST_PRIMITIVES[primitive]["expected"]

        assert observed == expected
        assert isinstance(observed, type(expected))
