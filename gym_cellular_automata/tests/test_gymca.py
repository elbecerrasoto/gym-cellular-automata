import gym

import gym_cellular_automata as gymca


def test_gymca():

    # Benchmark mode
    for env in gymca.envs:
        env = gym.make(env)
        assert isinstance(env, gym.Env)

    # Prototype mode
    for ProtoEnv in gymca.prototypes:
        env = ProtoEnv(nrows=42, ncols=42)
        assert isinstance(env, gym.Env)
