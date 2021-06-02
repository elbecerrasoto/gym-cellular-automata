import gym
from gym.spaces import Space

from gym_cellular_automata.forest_fire.bulldozer import ForestFireEnvBulldozerV1

# from matplotlib import pyplot as plt


THRESHOLD = 12


def test_gym_api():

    env = ForestFireEnvBulldozerV1()

    assert isinstance(env, gym.Env)
    assert isinstance(env.observation_space, Space)
    assert isinstance(env.action_space, Space)
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "render")
    assert hasattr(env, "close")
    assert hasattr(env, "seed")

    # Reset Step
    obs = env.reset()
    assert env.observation_space.contains(obs)
    # env.render(); plt.close()

    total_reward = 0.0
    done = False
    step = 0
    threshold = THRESHOLD

    # Random Policy for at most "threshold" steps
    while not done and step < threshold:
        step += 1
        # Step test
        action = env.action_space.sample()
        assert env.action_space.contains(action)

        obs, reward, done, info = env.step(action)
        # env.render(); plt.close()

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    env.close()
