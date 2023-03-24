import gymnasium as gym
import matplotlib
import pytest
from gymnasium.spaces import Space
from matplotlib import pyplot as plt

from gym_cellular_automata.registration import GYM_MAKE, REGISTERED_CA_ENVS

matplotlib.interactive(False)

STEPS = 3
RESETS = 2

# Plotting on each step is heavy on Bulldozer
PLOT_EACH = 256


@pytest.fixture
def envs():
    return (gym.make(env_call) for env_call in GYM_MAKE)


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

        for reset in range(resets):
            # Reset test
            obs = env.reset()
            assert env.observation_space.contains(obs)

            done = False
            step = 0

            max_steps = env.spec.max_episode_steps

            # Random Policy for at most "threshold" steps
            while not done and step < steps:
                if max_steps is not None:
                    if step > max_steps:
                        break

                step += 1
                # Step test
                action = env.action_space.sample()
                assert env.action_space.contains(action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Render test
                if step % plot_each == 0 or step <= 1:  # At least a couple of renders
                    assert isinstance(env.render(), matplotlib.figure.Figure)
                    plt.close("all")  # To garbage collect the figures

                assert env.observation_space.contains(obs)
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

        env.close()
