import gym
import numpy as np
import pytest
from gym.spaces import Space

from gym_cellular_automata.tests import MockCAEnv

STEPS = 8
RESETS = 8


@pytest.fixture
def env():
    return MockCAEnv()


def test_gym_api(env):
    assert isinstance(env, gym.Env)
    assert isinstance(env.observation_space, Space)
    assert isinstance(env.action_space, Space)
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "render")
    assert hasattr(env, "close")
    assert hasattr(env, "seed")


def test_step_reset(env):
    for reset in range(RESETS):
        obs = env.reset()
        assert env.observation_space.contains(obs)

        for step in range(STEPS):
            action = env.action_space.sample()
            assert env.action_space.contains(action)

            assert_step(env, env.step(action))


def assert_step(env, step_out):
    obs, reward, done, info = step_out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_gym_if_done_behave_gracefully(env):
    env.reset()
    env.done = True

    action = env.action_space.sample()

    with pytest.warns(UserWarning):
        step_out = obs, reward, done, info = env.step(action)

    assert_step(env, step_out)
    assert env.done
    assert reward == 0.0


def test_counts(env):
    obs = env.reset()
    grid, context = obs

    def get_dict_of_counts(grid):
        values, counts = np.unique(grid, return_counts=True)
        return dict(zip(values, counts))

    observed_counts = env.count_cells(grid)
    expected_counts = get_dict_of_counts(grid)

    assert all(
        [observed_counts[cell] == expected_counts[cell] for cell in expected_counts]
    )
