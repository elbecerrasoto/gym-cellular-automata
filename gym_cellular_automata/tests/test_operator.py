import pytest

from gym_cellular_automata import Operator

# Recyclable function for testing operators


@pytest.mark.skip(reason="WIP")
@pytest.fixture(scope="session")
def has_operator_specs():
    def has_operator_specs(operator) -> bool:
        """Test Atts, signatures and outputs"""
        ...
        return True

    return has_operator_specs


def operator_subclass():
    class OperatorSubclass(Operator):
        def __init__(*args, **kwargs):
            super.__init__(*args, **kwargs)

        def update(self, grid, action, context):
            return super.update(grid, action, context)

    return OperatorSubclass()


# Module testing
@pytest.mark.skip(reason="WIP")
def test_operator(operator_subclass, has_operator_specs):
    assert has_operator_specs(operator_subclass)
