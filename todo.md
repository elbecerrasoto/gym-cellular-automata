# To Do

## Short Term

- [ ] Reduce windy forest fire to three cells

  - The fourth cell is for visualization, but three cells are more elegant

- [ ] Support for _Python >=3.7_

  - Pinpoint dependencies versions, e.g. not working for _matplotlib  3.2.2_

- [ ] Fix the seed method

  - Right now is placebo
  - Add the corresponding tests
  - Test operator _deterministic attribute_

- [ ] Refactor _Envs globals_

  - To something like this
  ```python
  self._globalv = DEFAULT if globalv is None else globalv
  ```

- [ ] Add gifs of the _Envs_ in action

  - "A gif is worth a thousand images" Abraham Lincoln

- [ ] Polishing

  - Adopt practices from [hypermodern python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
  - Type annotations
  - Documentation

## Long Term

- [ ] Game of Life _CAEnv_

  - Implementation via scripting [Golly](http://golly.sourceforge.net/)

- [ ] :goggles: Any wild idea!

## Secondary

- [ ] Add common utility for training agents

  - One hot code encoding of the grid

- [ ] Profiling and optimization

  - C code and/or numba
