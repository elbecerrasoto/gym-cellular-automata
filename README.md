# Gym Automata
---

_Gym Automata_ is a collection of _Reinforcement Learning Environments_ (RLEs) that follow the [OpenAI Gym API](https://gym.openai.com/docs).

The available RLEs are based on [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) (CAs). On them an _Agent_ interacts with a CA, by changing its cell states, in a attempt to drive the emergent properties of its grid to a desired configuration.

## Installation

```bash
git clone https://github.com/elbecerrasoto/gym-cellular-automata
pip install -e gym-cellular-automata
```

## Basic Usage

### Random Policy
```python
import gym

env = gym.make("gym_cellular_automata:forest-fire-v0")
obs = env.reset()

total_reward = 0.0
done = False

# Random Policy
for i in range(12):
    if not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

print(f"Total Reward: {total_reward}")
```

### Available CA envs
```python
import gym_cellular_automata as gymca

# Print available CA envs
print(gymca.RESGISTERED_CA_ENVS)
```

## Gallery
![Forest Fire](pics/forest_fire.svg)

## Diagram
![Diagram](pics/gym_automata_diagram.svg)
