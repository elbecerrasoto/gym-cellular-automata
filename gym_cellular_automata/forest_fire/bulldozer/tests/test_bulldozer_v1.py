import gym
import pytest
from gym.spaces import Space

from gym_cellular_automata.forest_fire.bulldozer import ForestFireEnvBulldozerV1

# from matplotlib import pyplot as plt


THRESHOLD = 12


@pytest.mark.skip(reason="WIP")
def test_termination_behavior(env, all_trees):
    env.reset()

    env.grid = all_trees
    action = env.action_space.sample()

    # Acting on an all_tree grid causes termination
    obs, reward, done, info = env.step(action)
    grid, context = obs

    assert done


@pytest.mark.skip(reason="WIP")
def test_single_fire_seed(env):
    obs = env.reset()

    grid, context = obs
    ca_params, mod_params, freeze = context

    assert freeze == MAX_FREEZE

    # Single fire seed
    assert len(grid[grid == FIRE]) == 1
