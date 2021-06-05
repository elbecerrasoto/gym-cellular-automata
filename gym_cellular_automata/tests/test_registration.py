import gym
import matplotlib
import pytest
from gym.spaces import Space
from matplotlib import pyplot as plt

from gym_cellular_automata import REGISTERED_CA_ENVS

matplotlib.interactive(False)

LIBRARY = "gym_cellular_automata"

STEPS = 3
RESETS = 2

# Plotting on each step is heavy on Bulldozer
PLOT_EACH = 256


@pytest.fixture
def envs():
    return (gym.make(LIBRARY + ":" + ca_env) for ca_env in REGISTERED_CA_ENVS)


def are_operator_spaces_defined(operator):
    spaces = "grid_space", "action_space", "context_space"

    # Base Case. Empty Tuple
    if not operator.suboperators:
        for space in spaces:
            space = getattr(operator, space)
            if not isinstance(space, Space):
                return False
        return True

    for subop in operator.suboperators:

        if not are_operator_spaces_defined(subop):
            return False

    return True


@pytest.mark.skip(reason="Working on other stuff")
def test_operator_spaces(envs):
    for env in envs:
        assert are_operator_spaces_defined(env.MDP)


def test_gym_api_light(envs):
    assert_gym_api(envs, RESETS, STEPS, PLOT_EACH)


@pytest.mark.slow
def test_gym_api_heavy(envs):
    # Like ~10 min
    STEPS_HEAVY = 4096
    RESETS_HEAVY = 32
    assert_gym_api(envs, RESETS_HEAVY, STEPS_HEAVY, PLOT_EACH)


def assert_gym_api(envs, resets, steps, plot_each):

    for env in envs:

        assert isinstance(env, gym.Env)
        assert isinstance(env.observation_space, Space)
        assert isinstance(env.action_space, Space)
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "render")
        assert hasattr(env, "close")
        assert hasattr(env, "seed")

        for reset in range(resets):
            # Reset test
            obs = env.reset()
            assert env.observation_space.contains(obs)

            done = False
            step = 0

            # Random Policy for at most "threshold" steps
            while not done and step < steps:
                step += 1
                # Step test
                action = env.action_space.sample()
                assert env.action_space.contains(action)

                obs, reward, done, info = env.step(action)
                # Render test
                if step % plot_each == 0 or step <= 1:  # At least a couple of renders
                    assert isinstance(env.render(), matplotlib.figure.Figure)
                    plt.close("all")  # To garbage collect the figures

                assert env.observation_space.contains(obs)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)

        env.close()
