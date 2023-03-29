# To Do

## Short Term

- [x] [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) compatibility.

- [x] Reduce windy forest fire to three cells

- [ ] Fix the seed method, WORK IN PROGRESS
  - [x] _seed_ tested on _GridSpace_
  - [ ] _seed_ tested on _operators_
  - [ ] _seed_ tested on _CAEnv_
  - [ ] _seed_ tested at _registration_
  - [ ] Test operator _deterministic attribute_
  
- [ ] Release on _PyPI_ 
- [ ] Release on _conda-forge_

- [ ] Eliminate _gymnasium warnings_

- [x] Refactor _Envs globals_
- [x] Remove YAML

- [ ] Add pretty _gifs_ to [home page](README.md)

- [ ] Polishing
  - [ ] Update _docs_ & _contributing_
  - [ ] Adopt code style from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
  - [ ] Adopt practices from [hypermodern python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
  - [ ] Type annotations

## Long Term

- [ ] Game of Life _CAEnv_

  - Implementation via scripting [Golly](http://golly.sourceforge.net/)

- [ ] Check this library [cellpylib](https://github.com/lantunes/cellpylib)

- :goggles: Any wild idea!

## Secondary

- [ ] Add common utility for training agents

  - One hot code encoding of the grid

- [ ] Profiling and optimization

  - C code and/or numba

- [ ] ~~Support for _Python >=3.7_~~ NOT DOING

  - ~~Pinpoint dependencies versions, e.g. not working for _matplotlib  3.2.2_~~
