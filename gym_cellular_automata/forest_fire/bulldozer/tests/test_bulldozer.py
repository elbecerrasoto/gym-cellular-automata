import matplotlib
import pytest

from gym_cellular_automata.forest_fire.bulldozer import ForestFireBulldozerEnv
from gym_cellular_automata.grid_space import GridSpace

THRESHOLD = 12

NROWS, NCOLS = 256, 256


@pytest.fixture
def env():
    return ForestFireBulldozerEnv(nrows=NROWS, ncols=NCOLS, debug=True)


def test_bulldozerMDP_is_operator(env):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(env.MDP, strict=True)


def test_termination_behavior(env):
    env.reset()

    cells = env._empty, env._tree
    ncols, nrows = env.ncols, env.nrows
    non_fire = GridSpace(values=[cells], shape=(ncols, nrows)).sample()

    env.grid = non_fire
    action = env.action_space.sample()

    # Acting on an non-fire grid causes termination
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
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
