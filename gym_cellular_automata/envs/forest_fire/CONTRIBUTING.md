# Adding _Forest Fire CA Envs_

:evergreen_tree: :fire:

:man_firefighter: :woman_firefighter:

> How to add your own _Forest Fire CA Envs_?

## Steps

1. Set-Up
2. Code
3. Integrate

### Set-Up

+ Download and Install
```bash
  git clone https://github.com/elbecerrasoto/gym-cellular-automata
  cd gym-cellular-automata
  make install
```

+ Set-Up the development environment
```bash
  make develop
```

+ Check that everything is fine
```bash
  pytest
```

+ :octocat: Checkout a new development branch
```
  checkout -b my-awesome-env
```

+ Create a directory under `./forest_fire/`. There goes your _Forest Fire Env_. The name does not follow any convention, besides the appending of the version suffix `_v[0-9]`. Thus the directory name does not need to coincide with its final _gym_ registered name, it even could be a little cryptic (just document about it). 
```bash
  mkdir myAwesomeEnv_v0
```


### Code

+ Code your _Forest Fire Env_. You know what this means.
+  Pull and merge to get the latest hotness from _main_.
```bash
  git pull
  git checkout my-awesome-env
  git merge main
```
+ Read the following sections of this document.
  + [Directory Organization](#directory-organization)
  + [Design considerations](#design-considerations)
+ Now and then format your code, it is just _one simple command_.
    + ``` bash
        black .
      ```

### Integrate


#### Merge into _main_

> You should be merging into _main_ often, even if your _Forest Fire Env_ is somewhat functional.

1. Format the code
``` bash
black .
```
2. Test the code
``` bash
pytest
```
4. State your _dependencies_ on `setup.py`
5. Open a _pull request_
6. Wait for your changes to be accepted
7. :frog: Iterate all over again


#### Documentation

> Document your code.

:sleeping: :sleepy:

> Come on, it's worth it!

:penguin:

Document your environment on:
+ `./forest_fire/README.md` Brief description.
+ `./forest_fire/myAwesomeEnv_v0/myAwesomeEnv_v0.md` Full documentation.


#### Gym registration

Register your _Forest Fire Env_ on `./gym_cellular_automata/__init__.py`.

Args from _gym_ [registration.py](https://github.com/openai/gym/blob/master/gym/envs/registration.py):
+ **id** _(str)_: The official environment ID
+ **entry_point** _(Optional[str])_: The Python entrypoint of the environment class (e.g. module.name:Class)
+ **reward_threshold** _(Optional[int])_: The reward threshold before the task is considered solved
+ **nondeterministic** _(bool)_: Whether this environment is non-deterministic even after seeding
+ **max_episode_steps** _(Optional[int])_: The maximum number of steps that an episode can consist of
+ **kwargs** _(dict)_: The kwargs to pass to the environment class


The registration should look something like [this](https://github.com/openai/gym/blob/master/gym/envs/__init__.py):
```python
register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)
```


For inspiration on how to name your environment:
```python
# Print all gym registered envs
print("\n".join(vars(registry)["env_specs"]))
```


## Directory Organization

**Warning!** This only is an example of the directory organization. It does not reflect its current state. For simplicity `__init__.py` files are excluded.

+ `./forest_fire/`
    + `README.md` Brief description of each _Forest Fire Env_.
    + `CONTRIBUTING.md` This document.
    + `operators/` Common operators for all _Envs_. They could be specialized on each _Env_ directory.
        + `ca_drossel_schwabl.py` Original Drossel & Shawabl CA.
        + `ca_windy_convs.py` Convolution Implementation of Windy FF CA.
        + `mod_point.py` Parent class of all point-position modifiers.
        + `mod_bulldozer.py` Modifier for all _Bulldozer-like Envs_.
        + `coord_freeze.py` Coordinator by the _freeze_ parameter.
        + `coord_sequencer.py` Coordinator by _action-state_ timings.
        + `my_new_common_operator.py` A developed common operation.
    + `utils/` Common utility for all _Envs_. They could be specialized on each _Env_ directory.
        + `config.py` Common parser for global configurations.
        + `neighbors.py` Utility for getting cell neighborhoods.
        + `render_bulldozer.py` Shared utility for _Bulldozer-like Envs_.
    + `helicopter_v0/` A _Forest Fire Env_. 
        + `helicopter_v0.py` _Env_ code.
        + `helicopter_v0.yaml`  _Env_ global configuration.
        + `helicopter_v0.md` _Env_ documentation.
        + `operators/` _Env_ specialized operators.
            + `mod_helicopter.py` 
        + `utils/` _Env_ specialized utility.
            + `render.py`
            + `helicopter_shape.py`
    + `bulldozer_v0/` _Vanilla Bulldozer Env v0_.
        + `bulldozer_v0.py`
        + `bulldozer_v0.yaml`
        + `bulldozer_v0.md`
        + `utils/`
            + `config.py`
    + `bulldozer_v1/` _Vanilla Bulldozer Env v1_.
        + `bulldozer_v1.py` _Env_ code.
        + `bulldozer_v1.yaml`
        + `bulldozer_v1.md`
        + `utils/`
            + `render.py`
            + `config.py`
    + `bulldozerJumpyFlames_v0/` A _NDSL_ member's _Bulldozer Env_.
        + `bulldozerJumpyFlames_v0.py`
        + `bulldozerJumpyFlames_v0.yaml`
        + `bulldozerJumpyFlames_v0.md` Don't forget the docs :unamused:
        + `operators/`
            + `ca_jumpy.py`
            + `mod_bulldozerLP.py` A bulldozer with _Life Points_ or whatever!
        + `utils/`
            + `config.py`
            + `try_itNowOnTheAppStore.py`
            + `render.py`
    + `bulldozerJumpyFlames_v1` ... and so on and so forth ...
        + ...
    + `bulldozer_extraHard_v0` ... you get the point ...
        + ...


## Design considerations

:art: :brush:

### Operator Framework

You are by no means constrained to follow this, but at least you should consider it, as the _framework_ was built with this design choices.

> Everything is an Operator.

> The Environment layer is just _glue code_ for operators and utility.


### General architecture

1. **Layer 1** Code your favorite transformations for the underlying CA grid.
2. **Layer 2** Code an _operator_ to _coordinate_ your _layer 1_ transformations.
3. **Layer 3** Glue everything together (applying as few glue as possible) and put up a _facade_ to get a complete _gymEnv_.
4. You can be creative when combining and defining operators, like using several more layers.


### The Why

+ Putting everything as an _operator_ forces you to be _explicit_ about your state variables.
+ If all the variables are _explicit_, then only the _gymEnv_ last layer makes the decision of what or what not to make public to the user.
+ The tracking of _global state_ is delegated until the last layer. Delegation is good.
+ Putting emphasis on grid transformations allow us to define all the _MDP transitions_ independently of current state.
+ The _coordinator_ layer captures a single step transition, so each _step_ on the _gymEnv_ layer is one-to-one with the _update_ of the _coordinator_.
+ It provides a mental and formal model to thinking about _CA Envs_.
+ _Operators_ enable code recycling and modularity.
+ It defines a common API and enables collaboration.

Improvements or suggestions for the _architecture_ and _framework_ are always welcome, as development is on an early stage.

## Good Citizen Practices

+ Format your code, it is just _one simple command_.
  ``` bash
    black .
  ```

+ Do not break the build, easy check by running the test suite.
  ```bash
    pytest
  ```

+ Test your code, _it is good for your sanity_.
    + Testing framework: [pytest](https://docs.pytest.org/en/stable/)
    + I know that deadlines are always creeping in, so proceed with caution.
    + But testing always pays on the long run.
    + If you need to be convinced:
        1. Search for _Test Driven Development (TDD)_
        2. Necessary for the maturing of a code base.
        3. > "Good developers ship, great ones test."
