from gym_cellular_automata.operators import CellularAutomaton, Modifier, Coordinator
from gym_cellular_automata.tests.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Coordinator())

def test_Coordinator_API_specifications(coordinator = Coordinator()):
    assert coordinator.is_composition is True
    assert len(coordinator.suboperators) == 2
    assert isinstance(coordinator.suboperators[0], CellularAutomaton)
    assert isinstance(coordinator.suboperators[1], Modifier)
    assert hasattr(coordinator, 'cellular_automaton')
    assert hasattr(coordinator, 'modifier')
