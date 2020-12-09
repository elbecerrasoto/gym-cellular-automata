# Gym Automata
---

_Gym Automata_ is a collection of environments for Reinforcement Learning (RL), that follow the [OpenAI Gym API](https://gym.openai.com/docs).

The presented environments are based on [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) (CA).

A proper RL environment was defined by the addition of a _Modifier_ on top of the CA grid and CA dynamics.

The Modifier performs changes over the CA grid, according to the action taken by the _Agent_ hopefully driving the CA configuration to a desired goal.

## Gym Automata Interface

All the environments were built by the interplay of two main classes.
1. Automanton
2. Modifier

Both classes share the _update_ method. It operates over a CA grid. When the method is called by the automaton it calculates the next grid state following the automaton rules and neighborhood. When the method is called by the modifier some positions of the CA grid could changed accordingly to the environment semantics. For example a modifier operating over a CA that models epidemics would change grid cells from Suceptible or Infectious to Recovered, representing some intervention on the population. The interface of the _update_ method is (for the Automaton class action and state are set to None and are only there to maintain consistency.):
```python
obj.update(grid, action, state)
```

## Installation
1. Download and install 
```shell
git clone https://github.com/elbecerrasoto/gym-automata
pip install -e gym-automata
```
2. Import a _Cellular Automata based_ environment
```python
import gym
env = gym.make('gym_automata:minimal-example-v0')
```
