import gym

import gym_cellular_automata as gymca


def test_gymca():

    assert isinstance(gymca.envs, tuple) and isinstance(gymca.prototypes, tuple)

    # From a design perspective this is not necessarily true
    # For now is a serviceable test
    assert len(gymca.envs) == len(gymca.prototypes)

    # Benchmark mode
    for env in gymca.envs:
        env = gym.make(env)
        assert isinstance(env, gym.Env)

    # Prototype mode
    for ProtoEnv in gymca.prototypes:
        for rc in ((1, 1), (8, 8), (42, 42), (1024, 1024)):
            env = ProtoEnv(*rc)
            env.reset()
            assert isinstance(env, gym.Env)
