from gym_automata.builder_tools.operators import CellularAutomaton

from gym_automata.tests.builder_tools.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(CellularAutomaton())

def test_CellularAutomaton_API_specifications(cellular_automaton = CellularAutomaton()):
    assert cellular_automaton.is_composition is False
    assert cellular_automaton.suboperators == tuple()
