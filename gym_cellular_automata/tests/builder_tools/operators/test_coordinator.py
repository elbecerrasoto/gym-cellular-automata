from gym_automata.builder_tools.operators import CellularAutomaton
from gym_automata.builder_tools.operators import Modifier
from gym_automata.builder_tools.operators import Coordinator

from gym_automata.tests.builder_tools.operators.test_operator import test_Operator_API_specifications

test_Operator_API_specifications(Coordinator())

def test_Coordinator_API_specifications(coordinator = Coordinator()):
    assert coordinator.is_composition is True
    assert len(coordinator.suboperators) == 2
    assert isinstance(coordinator.suboperators[0], CellularAutomaton)
    assert isinstance(coordinator.suboperators[1], Modifier)
    assert hasattr(coordinator, 'cellular_automaton')
    assert hasattr(coordinator, 'modifier')
