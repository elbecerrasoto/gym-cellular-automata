import pytest

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.bulldozer import ForestFireEnvBulldozerV1

THRESHOLD = 12


@pytest.fixture
def env():
    return ForestFireEnvBulldozerV1()


def test_termination_behavior(env):
    env.reset()

    cells = env._empty, env._burned, env._tree
    ncols, nrows = env._col, env._row
    non_fire = GridSpace(values=[cells], shape=(ncols, nrows)).sample()

    env.grid = non_fire
    action = env.action_space.sample()

    # Acting on an non-fire grid causes termination
    obs, reward, done, info = env.step(action)
    grid, context = obs

    # Assert termination
    assert done


def test_starting_conditions_seed(env):
    obs = env.reset()

    grid, context = obs
    ca_params, mod_params, time_params = context

    # Internal clock at 0.0
    assert time_params == 0.0

    # Single fire seed
    assert len(grid[grid == env._fire]) == 1
