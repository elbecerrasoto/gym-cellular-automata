import gym
from gym_cellular_automata.envs.forest_fire import ForestFireEnv

from gym_cellular_automata.utils.config import get_forest_fire_config_dict
CONFIG = get_forest_fire_config_dict()

def test_forest_fire_env_specs(
                                env = ForestFireEnv()
                              ):
    
    assert isinstance(env, gym.Env)    
    assert hasattr(env, '_award')
    assert hasattr(env, '_is_done')
    assert hasattr(env, '_report')

# def test_forest_fire_step_output(
#                                     env = ForestFireEnv()
#                                 ):
#     env.step()
