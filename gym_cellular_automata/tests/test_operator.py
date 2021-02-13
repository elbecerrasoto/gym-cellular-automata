import inspect
from gym import spaces

from gym_cellular_automata import Operator


def test_Operator_API_specifications(operator=Operator()):

    assert isinstance(operator, Operator)

    if operator.is_composition is not None:
        assert isinstance(operator.is_composition, bool)

    assert isinstance(operator.suboperators, tuple)

    if operator.grid_space is not None:
        assert isinstance(operator.grid_space, spaces.Space)
    if operator.action_space is not None:
        assert isinstance(operator.action_space, spaces.Space)
    if operator.context_space is not None:
        assert isinstance(operator.context_space, spaces.Space)

    update_args = inspect.getfullargspec(operator.update).args
    assert update_args[1] == "grid"
    assert update_args[2] == "action"
    assert update_args[3] == "context"

    assert hasattr(operator, "__call__")
