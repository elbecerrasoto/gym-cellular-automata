from gym_cellular_automata import Operator
from gym_cellular_automata.tests import Identity


def test_operator():
    assert_operator(Identity(), strict=False)


def assert_operator(op, strict=False):
    from gym.spaces import Space

    def assert_optionals(obj, optional, atts, strict=False):
        """
        Asserting Optional Types
        """
        for att in atts:

            gatt = getattr(op, att)
            assert isinstance(gatt, optional) or gatt is None

            if strict:
                assert isinstance(
                    gatt, optional
                ), f"{att} expected to be {optional} or None./nHowever it is {type(gatt)}"

    def assert_update(op, strict=False):
        def block():
            grid = op.grid_space.sample()
            action = op.action_space.sample()
            context = op.context_space.sample()

            grid, context = op.update(grid, action, context)

            assert op.grid_space.contains(grid)
            assert op.context_space.contains(context)

        if strict:

            block()

        else:

            try:
                block()
            except AttributeError:
                pass

    assert isinstance(op, Operator)

    assert hasattr(op, "suboperators")
    assert isinstance(op.suboperators, tuple)

    for suop in op.suboperators:
        assert_operator(suop)

    assert_optionals(
        op, bool, ("grid_dependant", "action_dependant", "context_dependant"), strict
    )
    assert_optionals(op, Space, ("grid_space", "action_space", "context_space"), strict)

    assert_optionals(op, bool, ("deterministic",), strict)

    assert hasattr(op, "update")
    assert callable(op.update)

    assert_update(op, strict)
