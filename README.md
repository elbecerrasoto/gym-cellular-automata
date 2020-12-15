# Gym Automata
---

_Gym Automata_ is a collection of _Reinforcement Learning Environments_ (RLEs) that follow the [OpenAI Gym API](https://gym.openai.com/docs).

The available RLEs are based on [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) (CAs). On them an _Agent_ interacts with a CA, by changing its cell states, in a attempt to drive the emergent properties of its grid to a desired configuration.

## Gym Automata Interface

![Interface](pics/gym_automata_diagram.svg)

A _CA-based environment_ (CABE) can be made by a series of grid operations, those performed by a _Modifier_ and those performed by a CA local rules. If both operations are coordinated and extra functionality is added, like _reward_ calculation and _termination_ verification, a valid _RLE_ can be defined.

In order to provide a consistent and general way of CABE generation, _Gym Automata_ defines five types of Classes, which by functionality can be grouped into _Data_, _Operator_ and _Organizer_ Classes.
+ Data
	+ Grid
	+ MoState
+ Operator
	+ Automaton
	+ Modifier
+ Organizer
	+ CAEnv

The _Operator_ objects transform the grid in a series of operations which sequence is defined in a _CAEnv_ object. The grid is stored in a _Data_ object. Additionally some CABEs need to track the _Modifier_ state, which is captured in a _MoState_ object.

The _Operator_ objects can transform the grid by the shared method _update_. The _Automaton_ objects change the grid of a CA by computing a _1_-step update per cell, following the CA local function. The _Modifier_ objects change the grid at target cell positions accordingly to the taken _action_ and the current value of its own state.

+ _update_'s method syntax:
```python
grid = operator.update(grid, action, mostate)
```

The _update_ method is a pure function, so it explicitly needs to know the current _Modifier's state_, which is stored in the _MoState_ object. Thus the _CAEnv_ object is in charge of actively tracking its value.

The _update_ method of an _Automaton_ object does not depend on the _action_ and the _Modifier's_ state_, nonetheless, those arguments are in the method calling for consistency sake and are set to `None`.

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

## Minimal Example

Check a [minimal example](gym_automata/envs/minimal_example_env.py).
