import gymnasium as gym
import matplotlib
import numpy as np
import pytest
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.forest_fire.helicopter import ForestFireHelicopterEnv
from gym_cellular_automata.grid_space import GridSpace

RANDOM_POLICY_ITERATIONS = 12
TEST_GRID_ROWS = 3
TEST_GRID_COLS = 3

ROW, COL = 5, 5

up_left = 0
up = 1
up_right = 2
left = 3
not_move = 4
right = 5
down_left = 6
down = 7
down_right = 8

ACTION_NOT_MOVE = {not_move}

ACTION_LEFT = ({up_left, left, down_left},)
ACTION_RIGHT = ({up_right, right, down_right},)

ACTION_UP = ({up_left, up, up_right},)
ACTION_DOWN = ({down_left, down, down_right},)

REWARD_PER_EMPTY = 0.0
REWARD_PER_TREE = 1.0
REWARD_PER_FIRE = -1.0

REWARD_TYPE = TYPE_BOX

EMPTY = 0
TREE = 1
FIRE = 2

P_FIRE = 0.033
P_TREE = 0.333


@pytest.fixture
def env():
    return ForestFireHelicopterEnv(ROW, COL, debug=True)


@pytest.fixture
def all_fire_grid():
    return GridSpace(
        values=[EMPTY, TREE, FIRE],
        shape=(TEST_GRID_ROWS, TEST_GRID_COLS),
        probs=[0.0, 0.0, 1.0],
    ).sample()


def set_env_with_custom_state(env, grid, context):
    env.grid = grid
    env.context = context
    return env


@pytest.fixture(scope="module")
def reward_space():
    reward_weights = np.array([REWARD_PER_EMPTY, REWARD_PER_TREE, REWARD_PER_FIRE])

    max_weight = np.max(reward_weights)
    min_weight = np.min(reward_weights)

    lower_bound = np.array(ROW * COL * min_weight)
    upper_bound = np.array(ROW * COL * max_weight)

    return spaces.Box(lower_bound, upper_bound, dtype=TYPE_BOX)


def test_forest_fire_env_is_a_gym_env(env):
    assert isinstance(env, gym.Env)


def test_helicopterMDP_is_operator(env):
    from gym_cellular_automata.tests import assert_operator

    assert_operator(env.MDP, strict=True)


def test_forest_fire_env_private_methods(env, reward_space):
    env.reset()
    action = env.action_space.sample()
    env.step(action)

    assert hasattr(env, "_award")
    reward = np.array(env._award(), dtype=TYPE_BOX)
    assert reward_space.contains(reward)

    assert hasattr(env, "_is_done")
    assert isinstance(env._is_done(), bool)

    assert hasattr(env, "_report")
    assert isinstance(env._report(), dict)


def test_forest_fire_env_step_output(env):
    env.reset()
    action = env.action_space.sample()
    gym_api_out = env.step(action)

    assert isinstance(gym_api_out, tuple)
    assert len(gym_api_out) == 5

    assert isinstance(gym_api_out[1], float)
    assert isinstance(gym_api_out[2], bool)
    assert isinstance(gym_api_out[4], dict)


def test_forest_fire_env_output_spaces(env, reward_space):
    obs, info = env.reset()

    assert env.observation_space.contains(obs)

    grid, (ca_params, pos, freeze) = obs
    assert env.grid_space.contains(grid)
    assert env.ca_params_space.contains(ca_params)
    assert env.position_space.contains(pos)
    assert env.freeze_space.contains(freeze)


def assert_observation_and_reward_spaces(env, obs, reward, reward_space):
    reward = np.array(reward, dtype=TYPE_BOX)

    assert env.observation_space.contains(obs)
    assert reward_space.contains(reward)

    grid, (ca_params, pos, freeze) = obs

    assert env.grid_space.contains(grid)
    assert env.ca_params_space.contains(ca_params)
    assert env.position_space.contains(pos)
    assert env.freeze_space.contains(freeze)

    # Strong typing of the observations
    assert isinstance(grid, np.ndarray)
    assert isinstance(ca_params, np.ndarray)
    assert isinstance(pos, np.ndarray)
    assert isinstance(freeze, np.ndarray)


def test_forest_fire_env_with_random_policy(env, reward_space):
    env.reset()

    for step in range(RANDOM_POLICY_ITERATIONS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert_observation_and_reward_spaces(env, obs, reward, reward_space)


def test_forest_fire_env_hit_info(env, all_fire_grid):
    def assert_hit_right_notmove_down():
        obs, reward, terminated, truncated, info = env.action(ACTION_RIGHT)

        assert info["hit"] is True

        obs, reward, terminated, truncated, info = env.action(ACTION_NOT_MOVE)

        assert info["hit"] is False

        obs, reward, terminated, truncated, info = env.action(ACTION_DOWN)

        assert info["hit"] is True


def test_env_render(env):
    env.reset()
    assert isinstance(env.render(), matplotlib.figure.Figure)


def manual_assesment(verbose=False):
    from time import sleep

    env = ForestFireHelicopterEnv()

    obs, info = env.reset()

    if verbose:
        print(f"\n\nThe FIRST obs is {obs}")

    env.render()

    for i in range(64):
        action = env.action_space.sample()
        if verbose:
            print(f"\n\nAction Selected: {action}")

        obs, reward, _, info = env.step(action)

        env.render()

        if verbose:
            print("\nObservation")
            print(f"Obs: {obs}")
            print(f"Reward: {reward}")
            print(f"Info: {info}\n")

        sleep(0.2)
        print(".", end="")
