import pytest  # With plugin pytest-repeat

import gym
from gym import spaces

from gym_cellular_automata.envs.forest_fire_v1 import ForestFireEnv

from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]


EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


MAX_FREEZE = CONFIG["max_freeze"]


@pytest.fixture
def env():
    return ForestFireEnv()


@pytest.fixture
def grid_space():
    return Grid(values=[EMPTY, TREE, FIRE], shape=(ROW, COL))


@pytest.fixture
def ca_params_space():
    return spaces.Box(0.0, 1.0, shape=(3, 3))


@pytest.fixture
def mod_params_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.fixture
def coord_params_space():
    return spaces.Discrete(MAX_FREEZE + 1)


@pytest.fixture
def context_space(ca_params_space, mod_params_space, coord_params_space):
    return spaces.Tuple((ca_params_space, mod_params_space, coord_params_space))


def test_forest_fire_env_is_a_gym_env(env, grid_space, context_space):
    assert isinstance(env, gym.Env)


@pytest.mark.repeat(2)
def test_gym_reset(env, grid_space, context_space):

    grid, context = env.reset()

    assert not env.done

    assert grid_space.contains(grid)
    assert context_space.contains(context)


@pytest.mark.repeat(2)
def test_gym_step(env, grid_space, context_space):

    max_steps = 8

    env.reset()

    step = 0

    while env.done and step < max_steps:

        step += 1

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, tuple)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        grid, context = obs

        assert grid_space.contains(grid)
        assert context_space.contains(context)


def test_gym_if_done_behave_gracefully(env, grid_space, context_space):
    env.reset()
    env.done = True

    action = env.action_space.sample()
    with pytest.warns(UserWarning):
        obs, reward, done, info = env.step(action)

    assert done
    assert reward == 0.0
