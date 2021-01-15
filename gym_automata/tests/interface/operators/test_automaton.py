from gym_automata.interface.operators.automaton import Automaton
from gym_automata.tests.interface.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Automaton())

def test_Automaton_API_specifications(automaton = Automaton()):
    assert automaton.is_composition is False
    assert automaton.suboperators == tuple()
