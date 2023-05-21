# Releases

:drum:


## 0.6.0

> Theme: New API, gymnasium compatibility

+ New API
  + `reset` now returns `obs, info`
  + `step` now returns `obs, reward, terminated, truncated, info`
  + Rename of some attributes for the new API
  + Deletion of old API unused code
+ _Work In Progress_, make the seed for RNG functional
+ Fix `matplotlib` warnings


## 0.5.6

> Theme: Remove YAML globals

+ Simplification of globals
  + Import parameters into _env init method_
  + Globals are read directly, instead from config files
  + Remove dependency on YAML
  + Simplification of _CAEnv ABC_
+ Fix definition of bulldozer to any grid size _[1-n]_


## 0.5.5

> Theme: Prototype & Benchmark modes

+ Prototype & Benchmark mode.
    + _gymca_ can be called in two ways
    + `gymca.prototypes` & `gymca.envs`
+ New _env_ version  "ForestFireBulldozer256xx256-v3"
    + Windy Forest Fire was rolled back to 3 cells
+ `__init__.py` clean up


## 0.5.4

> Theme: Update to _gym 0.21.0_

+ Casting observations to _float32_ to pass tests
  + _space.Box.contains_ returns _False_ when comparing different float types
+ _To Do List_ file to help development
+ Refactor to input _rows_ and _cols_ on registration code
  + Specification of _rows_ an _cols_ removed from _yaml files_
+ `gym_cellular_automata.__version__` variable
  + Dropping the _v_ from the version number e.g. _0.5.4_ instead of _v0.5.4_
+ Render title refactors
  + Different titles for manually created and registered environments
+ Soft linking to find the _CAEnvs_ on expected location
  + envs directory
+ Debug flag on CAEnv base class


## 0.5.3

> Theme: Environment difficulty level via _nrows_ & _ncols_

+ All environments now depend on the number of rows and columns
  + This functions as a proxy for varying levels of difficulty
+ **CAEnv ABC** changes
  + _Status method_
  + Utility for defaults (subject to change)
+ Adding of Environments
  + _ForestFireHelicopter5x5-v1_
  + _ForestFireBulldozer256x256-v2_
+ Deletion of Environments
  + _ForestFireHelicopter-v0_
  + _ForestFireBulldozer-v1_
+ Operator test
+ Hotfix: gym seeding utility (*np_random* instead of *np.random*)
+ Registration code maintenance


## 0.5.2

> Theme: Bulldozer Render

+ Improved Bulldozer Render
  + Better categorization (Tree vs Non-Tree) of cell counts bar plot
  + Better support for different sizes of grid
  + Automatic align of *location* and fire *markers*
+ Minor fix on *Move Operator*
+ Architecture documentation (WIP)
+ Maintenance
  + Removed dependencies
  + Refactored shared render utility
  + *update_gallery* script improved
