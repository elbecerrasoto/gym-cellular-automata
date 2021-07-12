import gym
import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.helicopter import ForestFireEnvHelicopterV0
from gym_cellular_automata.forest_fire.helicopter.utils.config import CONFIG

RANDOM_POLICY_ITERATIONS = 12
TEST_GRID_ROWS = 3
TEST_GRID_COLS = 3

ROW = CONFIG["grid_shape"]["nrows"]
COL = CONFIG["grid_shape"]["nrows"]

ACTION_NOT_MOVE = CONFIG["actions"]["not_move"]

ACTION_LEFT = CONFIG["actions"]["left"]
ACTION_RIGHT = CONFIG["actions"]["right"]

ACTION_UP = CONFIG["actions"]["up"]
ACTION_DOWN = CONFIG["actions"]["down"]

REWARD_PER_EMPTY = CONFIG["rewards"]["per_empty"]
REWARD_PER_TREE = CONFIG["rewards"]["per_tree"]
REWARD_PER_FIRE = CONFIG["rewards"]["per_fire"]

REWARD_TYPE = np.float32

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

P_FIRE = CONFIG["ca_params"]["p_fire"]
P_TREE = CONFIG["ca_params"]["p_tree"]


@pytest.fixture
def env():
    return ForestFireEnvHelicopterV0()


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

    lower_bound = np.array(ROW * COL * min_weight, dtype=REWARD_TYPE)
    upper_bound = np.array(ROW * COL * max_weight, dtype=REWARD_TYPE)

    return spaces.Box(lower_bound, upper_bound)


def test_forest_fire_env_is_a_gym_env(env):
    assert isinstance(env, gym.Env)


def test_forest_fire_env_private_methods(env, reward_space):
    env.reset()
    action = env.action_space.sample()
    env.step(action)

    assert hasattr(env, "_award")
    assert reward_space.contains(env._award())

    assert hasattr(env, "_is_done")
    assert isinstance(env._is_done(), bool)

    assert hasattr(env, "_report")
    assert isinstance(env._report(), dict)


def test_forest_fire_env_step_output(env):
    env.reset()
    action = env.action_space.sample()
    gym_api_out = env.step(action)

    assert isinstance(gym_api_out, tuple)
    assert len(gym_api_out) == 4

    assert isinstance(gym_api_out[1], float)
    assert isinstance(gym_api_out[2], bool)
    assert isinstance(gym_api_out[3], dict)


def test_forest_fire_env_output_spaces(env, reward_space):
    obs0 = env.reset()

    assert env.observation_space.contains(obs0)

    grid, (ca_params, pos, freeze) = obs0
    assert env.grid_space.contains(grid)
    assert env.ca_params_space.contains(ca_params)
    assert env.position_space.contains(pos)
    assert env.freeze_space.contains(freeze)


def assert_observation_and_reward_spaces(env, obs, reward, reward_space):
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
        obs, reward, done, info = env.step(action)

        assert_observation_and_reward_spaces(env, obs, reward, reward_space)


def test_forest_fire_env_hit_info(env, all_fire_grid):
    def assert_hit_right_notmove_down():
        obs, reward, done, info = env.action(ACTION_RIGHT)

        assert info["hit"] is True

        obs, reward, done, info = env.action(ACTION_NOT_MOVE)

        assert info["hit"] is False

        obs, reward, done, info = env.action(ACTION_DOWN)

        assert info["hit"] is True


def manual_assesment(verbose=False):
    from time import sleep

    env = ForestFireEnvHelicopterV0()

    obs = env.reset()

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
