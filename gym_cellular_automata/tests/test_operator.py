from gym_cellular_automata import Operator
from gym_cellular_automata.tests import Identity


def test_operator():
    assert_operator(Identity())


def assert_operator(op):
    from gym.spaces import Space

    def assert_optionals(obj, optional_type, *atts):
        for att in atts:
            assert hasattr(op, att)
            if getattr(op, att) is not None:
                assert isinstance(getattr(op, att), optional_type)

    def assert_update(op):
        try:

            grid = op.grid_space.sample()
            action = op.action_space.sample()
            context = op.context_space.sample()

            grid, context = op.update(grid, action, context)

            assert op.grid_space.contains(grid)
            assert op.context_space.contains(context)

        except AttributeError:
            pass

    assert isinstance(op, Operator)

    assert hasattr(op, "suboperators")
    assert isinstance(op.suboperators, tuple)

    from icecream import ic
    from objprint import objprint

    objprint(op)
    ic(op.suboperators)

    for suop in op.suboperators:
        assert suop is not tuple()

    for suop in op.suboperators:
        assert_operator(suop.suboperators)

    assert_optionals(
        op, bool, "grid_dependant", "action_dependant", "context_dependant"
    )
    assert_optionals(op, Space, "grid_space", "action_space", "context_space")
    assert_optionals(op, bool, "deterministic")

    assert hasattr(op, "update")
    assert callable(op.update)

    assert_update(op)
