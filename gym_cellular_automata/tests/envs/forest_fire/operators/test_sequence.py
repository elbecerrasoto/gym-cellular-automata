"""
import pytest
import numpy as np

from gym_cellular_automata.grid_space import Grid

from gym_cellular_automata.envs.forest_fire.v2.operators.sequencer import Sequencer

from gym_cellular_automata.envs.forest_fire.v1.operators import Bulldozer
from gym_cellular_automata.envs.forest_fire.v1.operators import WindyForestFireB

from gym_cellular_automata.envs.forest_fire.v1.utils.config import CONFIG


@pytest.fixture
def ca():
    return WindyForestFireB


# Deterministic Wind
# Makes the CA update deterministic
@pytest.fixture
def wind(ca):
    return ca.context_space.high


@pytest.fixture
def bulldozer():
    return Bulldozer()



@pytest.fixtures
def operators(ca, bulldozer):
    return np.array([ca, bulldozer])


@pytest.fixture
def operators_contexts():
    pass


@pytest.fixture
def sequencer(operators):
    return Sequencer(operators)


@pytest.fixture
def apply_operation_flow_callable(sequencer):
    return sequencer()


@pytest.mark.parametrize(
    "operation_sequence",
    [
     [0], [1], [0, 1],
     [0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 1, 0, 1],
     [1, 0, 1, 0]
    ]
)
def test_manual_sequencing_vs_via_sequencer(sequencer):
    pass



# Key test
def test_manual_test_vs_automatic():
    pass


def test_grid_space():
    pass


def test_ca_units():
    pass




# White box test
def operation_flow_rets_idxs():
    pass


def test_grid_context_spaces():
    pass

# More white testing
def test_dict_inits():
    pass
"""
