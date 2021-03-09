from gym_cellular_automata.envs.forest_fire_v1.forest_fire_v1 import ForestFireEnv


env = ForestFireEnv()

env.reset()


for i in range(66):
    action = env.action_space.sample()
    print(env.step(action))
