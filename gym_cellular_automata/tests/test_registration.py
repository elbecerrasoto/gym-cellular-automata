import gym
import matplotlib
import pytest
from gym.spaces import Space
from matplotlib import pyplot as plt

from gym_cellular_automata import REGISTERED_CA_ENVS

matplotlib.interactive(False)

LIBRARY = "gym_cellular_automata"

STEPS = 2
RESETS = 2


def test_light():
    assert_gym_api(RESETS, STEPS)


@pytest.mark.slow
def test_soak():
    # Like ~4hrs
    STEPS_HEAVY = 1024
    RESETS_HEAVY = 64
    assert_gym_api(RESETS_HEAVY, STEPS_HEAVY)


def assert_gym_api(resets, steps):
    for ca_env in REGISTERED_CA_ENVS:

        calling_string = LIBRARY + ":" + ca_env
        env = gym.make(calling_string)

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

            # Step test
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
                assert isinstance(env.render(), matplotlib.figure.Figure)
                plt.close("all")  # To garbage collect the figures

                assert env.observation_space.contains(obs)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)

        env.close()
