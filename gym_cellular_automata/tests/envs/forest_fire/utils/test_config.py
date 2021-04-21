import pytest

from gym_cellular_automata.envs.forest_fire.utils.config import (
    get_config_dict,
    group_actions,
    translate,
)

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


class TestCommonParsings:
    @pytest.fixture
    def mapping(self):
        L = 12
        return {key: key for key in range(L)}

    @pytest.fixture
    def anames(self):
        return (
            "up_left",
            "up",
            "up_right",
            "left",
            "not_move",
            "right",
            "down_left",
            "down",
            "down_right",
        )

    @pytest.fixture
    def anames_per_group(self, anames):
        return {
            "up": (anames[0], anames[1], anames[2]),
            "down": (anames[6], anames[7], anames[8]),
            "left": (anames[0], anames[3], anames[6]),
            "right": (anames[2], anames[5], anames[8]),
            "not_move": (anames[4],),
        }

    @pytest.fixture
    def actions(self, anames):
        return {aname: action for action, aname in enumerate(anames)}

    def test_translate(self, mapping):
        f = str
        translation = {key: f(key) for key in mapping}
        observed = translate(mapping, translation)
        expected = {f(key): f(key) for key in observed}

        assert all([observed[key] == expected[key] for key in observed])

    def test_group_actions(self, anames, anames_per_group, actions):
        grouped_actions = group_actions(actions)

        all_in_group = lambda akeys, group: all(
            [actions[akey] in group for akey in akeys]
        )

        for group_name in anames_per_group:
            assert all_in_group(
                anames_per_group[group_name], grouped_actions[group_name]
            )
