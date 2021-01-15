from gym_automata.interface.operators import Exchanger

from gym_automata.tests.interface.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Exchanger())

def test_Exchanger_API_specifications(exchanger = Exchanger()):
    assert exchanger.is_composition is False
    assert exchanger.suboperators == tuple()
    assert hasattr(exchanger, 'effects')
