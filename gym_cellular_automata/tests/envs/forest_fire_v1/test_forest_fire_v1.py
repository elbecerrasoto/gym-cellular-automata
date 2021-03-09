from gym_cellular_automata.envs.forest_fire_v1.forest_fire_v1 import ForestFireEnv

REPS = 2
MAX_STEPS = 1024

env = ForestFireEnv()
for episode in range(REPS):

    env.reset()
    done = False
    
    i = 0

    print(f"\nSTART {episode} EPISODE")

    while not done and i < MAX_STEPS:
        i += 1
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if i % 24 == 0:
            print(f"episode: {episode}, step: {i}")
            print(f"obs (context): {obs[1][1:]} reward: {reward}")
