# Gym Automata
---

_Gym Automata_ is a collection of _Reinforcement Learning Environments_ (RLEs) that follow the [OpenAI Gym API](https://gym.openai.com/docs).

The available RLEs are based on [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) (CAs).

A proper RLE was defined by the addition of a _Modifier_ on top of a CA.

The _Modifier_ performs changes over the CA grid according to the action taken by the _Agent_ trying to drive the CA configuration to the desired goal.

## Gym Automata Interface

![Interface](pics/gym_automata_diagram.svg)

A _CA-based environment_ (CABE) can be thought of as a series of grid operations, those performed by the _Modifier_ and those performed by the _Automaton_. An upper layer controls the operation order and gives them the semantics of an RLE, such as _reward_ calculation and _termination_ verification.

This is accomplished by abstracting the operations into the _Operator_ objects and the upper layer into a _Wrapper_ object. The grid being transformed is codified into a _Data_ object. Some CABEs need to track the _Modifier_ state, which is codified into a _State_ object.

The following objects are used to define a CABE:
+ Data objects
	+ Grid
	+ State
+ Operator objects
	+ Automaton
	+ Modifier
+ Wrapper objects
	+ CAEnv

All the environments are built by the interplay of two _operator_ classes.
+ Automaton
+ Modifier

The _operator_ objects can transform the grid (configuration) of a CA by the shared method _update_.

```python
grid = operator.update(grid, action, state)
```

The _Automaton_ objects change the grid of a CA by computing a _1_-step update per cell, following a local function (CA rules and neighborhood).

The _Modifier_ objects change the grid at target cell positions accordingly to the taken _action_ and its _state_. The _Modifier_ represents the control task being carried on top of the CA.

Those two operations over the grid are abstracted into the _update_ method. The _update_ method is a pure function, which explicitly needs to know the current state of the _Modifier_.

The _update_ of an _Automaton_ does not depends on _action_ and _state_, nonetheless, those arguments are still provided to make it consistent across the library.

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
