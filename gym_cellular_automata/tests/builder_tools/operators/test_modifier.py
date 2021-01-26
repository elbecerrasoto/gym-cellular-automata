from gym_cellular_automata.builder_tools.operators import Modifier
from gym_cellular_automata.tests.builder_tools.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Modifier())

def test_Modifier_API_specifications(modifier = Modifier()):
    assert modifier.is_composition is False
    assert modifier.suboperators == tuple()
    assert hasattr(modifier, 'effects')
