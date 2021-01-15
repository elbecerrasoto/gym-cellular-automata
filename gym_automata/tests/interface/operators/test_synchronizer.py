from gym_automata.interface.operators import Automaton
from gym_automata.interface.operators import Exchanger
from gym_automata.interface.operators import Synchronizer

from gym_automata.tests.interface.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Synchronizer())

def test_synchronizer_API_specifications(synchronizer = Synchronizer()):
    assert synchronizer.is_composition is True
    assert len(synchronizer.suboperators) == 2
    assert isinstance(synchronizer.suboperators[0], Automaton)
    assert isinstance(synchronizer.suboperators[1], Exchanger)
    assert hasattr(synchronizer, 'automaton')
    assert hasattr(synchronizer, 'exchanger')
