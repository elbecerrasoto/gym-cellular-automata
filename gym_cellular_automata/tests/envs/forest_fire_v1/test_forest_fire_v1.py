import pytest  # With plugin pytest-repeat
import numpy as np

import gym
from gym import spaces

from gym_cellular_automata.envs.forest_fire_v1 import ForestFireEnv

from gym_cellular_automata.grid_space import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG


ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

MAX_FREEZE = CONFIG["max_freeze"]

REWARD_PER_TREE = CONFIG["rewards"]["per_tree"]


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


@pytest.fixture
def all_trees():
    return Grid(
        values=[EMPTY, TREE, FIRE], shape=(ROW, COL), probs=[0.0, 1.0, 0.0]
    ).sample()


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


def test_termination_behavior(env, all_trees):
    env.reset()

    env.grid = all_trees
    action = env.action_space.sample()

    # Acting on an all_tree grid causes termination
    obs, reward, done, info = env.step(action)
    grid, context = obs

    assert done

    def get_dict_of_counts(grid):
        values, counts = np.unique(grid, return_counts=True)
        return dict(zip(values, counts))

    assert reward == REWARD_PER_TREE * get_dict_of_counts(grid)[TREE]


def test_single_fire_seed(env):
    obs = env.reset()

    grid, context = obs
    ca_params, mod_params, freeze = context

    assert freeze == MAX_FREEZE

    # Single fire seed
    assert len(grid[grid == FIRE]) == 1


def manual_assesment(steps=2048, verbose=False, wait=0.1, simulate_ff=False):
    from time import sleep

    done = False

    if simulate_ff:
        ForestFireEnv._max_freeze = 0

    env = ForestFireEnv()

    obs = env.reset()

    grid, context = obs
    wind, pos, freeze = context

    for step in range(steps):
        if done:
            break

        if verbose:
            print(f"\n\n------------ Step: {step} ------------")
            print(f"Grid:\n{grid}\n")
            print(f"Wind:\n{wind}\n")
            print(f"Position:\n{pos}\n")
            print(f"Freeze:\n{freeze}\n")

        env.render()

        action = env.action_space.sample()

        if verbose:
            print(f"Selected Action:\n{action}")

        obs, reward, done, info = env.step(action)

        grid, context = obs
        wind, pos, freeze = context

        if not verbose:
            print(".", end="")
            if step % 8 == 0:
                print(step, end="")
        sleep(wait)
