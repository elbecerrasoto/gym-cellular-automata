import gym
import matplotlib
import pytest
from gym.spaces import Space
from matplotlib import pyplot as plt

from gym_cellular_automata import GYM_MAKE, REGISTERED_CA_ENVS

matplotlib.interactive(False)

STEPS = 3
RESETS = 2

# Plotting on each step is heavy on Bulldozer
PLOT_EACH = 256


@pytest.fixture
def envs():
    return (gym.make(env_call) for env_call in GYM_MAKE)


# @pytest.mark.skip(reason="debugging")
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


def are_operator_spaces_defined(operator):
    spaces = "grid_space", "action_space", "context_space"
    from objprint import objprint as obp

    def check_op(operator):

        for space in spaces:
            space = getattr(operator, space)

            if not isinstance(space, Space):
                obp(operator)  # Debug print
                return False

        return True

    # Base Case. Empty Tuple
    if not operator.suboperators:
        return check_op(operator)

    else:

        for subop in operator.suboperators:

            current_level = check_op(subop)
            lower_level = are_operator_spaces_defined(subop)

            if not (current_level and lower_level):
                return False

        return True


def test_are_operator_spaces_defined():
    def generate_deep_suboperators(spaces_defined=True):
        from gym import spaces

        from gym_cellular_automata import GridSpace
        from gym_cellular_automata.tests import Identity

        gS = GridSpace(values=[55, 66, 77], shape=(2, 2))
        aS = spaces.Discrete(2)
        cS = spaces.Discrete(3)

        I = lambda: Identity(gS, aS, cS)
        main = I()

        deep_call = I if spaces_defined else Identity

        Li = list()
        lj = list()
        for i in range(2):
            for j in range(3):
                lj.append(deep_call())
            tmp = I()
            tmp.suboperators = tuple(lj)
            Li.append(tmp)
        main.suboperators = tuple(Li)
        return main

    epass = generate_deep_suboperators(spaces_defined=True)
    efail = generate_deep_suboperators(spaces_defined=False)
    assert are_operator_spaces_defined(epass)
    assert not are_operator_spaces_defined(efail)
