import gym
from gym.spaces import Space

from gym_cellular_automata import REGISTERED_CA_ENVS

LIBRARY = "gym_cellular_automata"


def test_gym_api():
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

        # Reset test
        obs = env.reset()
        assert env.observation_space.contains(obs)

        # Step test
        action = env.action_space.sample()
        assert env.action_space.contains(action)

        obs, reward, done, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        env.close()
