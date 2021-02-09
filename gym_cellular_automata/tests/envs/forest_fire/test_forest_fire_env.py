import pytest
import numpy as np

import gym
from gym import spaces

import gym_cellular_automata
from gym_cellular_automata.envs.forest_fire import ForestFireEnv
from gym_cellular_automata.envs.forest_fire.utils.config import get_forest_fire_config_dict

RANDOM_POLICY_ITERATIONS = 12

CONFIG = get_forest_fire_config_dict()

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

REWARD_PER_EMPTY = CONFIG['rewards']['per_empty']
REWARD_PER_TREE  = CONFIG['rewards']['per_tree']
REWARD_PER_FIRE  = CONFIG['rewards']['per_fire']

@pytest.fixture
def env():
    env = ForestFireEnv()
    return env

@pytest.fixture(scope = 'module')
def reward_space():
    reward_weights = np.array([REWARD_PER_EMPTY,
                               REWARD_PER_TREE,
                               REWARD_PER_FIRE])
    
    max_weight = np.max(reward_weights)
    min_weight = np.min(reward_weights)

    lower_bound = np.array(ROW * COL * min_weight, dtype=np.float32)
    upper_bound = np.array(ROW * COL * max_weight, dtype=np.float32)
    
    return spaces.Box(lower_bound, upper_bound)

def test_forest_fire_env_is_a_gym_env(env):
    assert isinstance(env, gym.Env)

def test_forest_fire_env_private_methods(env, reward_space):
    env.reset()

    assert hasattr(env, '_award')
    assert reward_space.contains(env._award())
    
    assert hasattr(env, '_is_done')
    assert isinstance(env._is_done(), bool)
    
    assert hasattr(env, '_report')
    assert isinstance(env._report(), dict)

def test_forest_fire_key_attributes(env):
    env.reset()
    hasattr(env, 'grid')
    isinstance(env.grid, np.ndarray)
    hasattr(env, 'context')
    hasattr(env, 'coordinator')
    isinstance(env.coordinator, gym_cellular_automata.Operator)

def test_forest_fire_env_reset_output(env):
    gym_api_out = env.reset()
    
    assert isinstance(gym_api_out, tuple)
    assert len(gym_api_out) == 4
    
    assert isinstance(gym_api_out[1], float)
    assert isinstance(gym_api_out[2], bool)
    assert isinstance(gym_api_out[3], dict)

def test_forest_fire_env_output_spaces(env, reward_space):
    obs0, reward, done, info = env.reset()
    
    assert reward_space.contains(reward)
    assert env.observation_space.contains(obs0)

    grid, (ca_params, pos, freeze) = obs0   
    assert env.grid_space.contains(grid)
    assert env.ca_params_space.contains(ca_params)
    assert env.pos_space.contains(pos)
    assert env.freeze_space.contains(freeze)

def assert_observation_and_reward_spaces(env, obs, reward, reward_space):
    assert env.observation_space.contains(obs)
    assert reward_space.contains(reward)
    
    grid, (ca_params, pos, freeze) = obs

    assert env.grid_space.contains(grid)
    assert env.ca_params_space.contains(ca_params)
    assert env.pos_space.contains(pos)
    assert env.freeze_space.contains(freeze)
    
    # Strong typing of the observations
    assert isinstance(grid, np.ndarray)
    assert isinstance(ca_params, np.ndarray)
    assert isinstance(pos, np.ndarray)
    assert isinstance(freeze, np.ndarray)
    
def test_forest_fire_env_with_random_policy(env, reward_space):
    obs, reward, done, info = env.reset()
    
    for step in range(RANDOM_POLICY_ITERATIONS):
        
        assert_observation_and_reward_spaces(env, obs, reward, reward_space)
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
def manual_assesment(verbose = False):
    from time import sleep
    env=ForestFireEnv()
    
    obs, reward, _, _ = env.reset()
    
    if verbose:
        print(f'\n\nThe FIRST obs is {obs}')
        print(f'With reward {reward}')
        
    env.render()
    
    for i in range(64):

        action = env.action_space.sample()
        if verbose:
            print(f'\n\nAction Selected: {action}')
        
        obs, reward, _, _ = env.step(action)

        env.render()
        
        if verbose:
            print('\nObservation')
            print(f'Obs: {obs}')
            print(f'Reward: {reward}\n')
        
        sleep(0.2)
        print('.', end='')
