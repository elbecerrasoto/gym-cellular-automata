# Architecture #

Architecture of _gym cellular automata library_ (GYMCA).

---

## Introduction

The building of a _Cellular Automata Environment_ (CAE) is accomplished by defining operations over the *Cellular Automata* (CA) grid and then giving those operations the semantics of a *Reinforcement Learning* (RL) task.

For example the calculation of the CA rules could be one operation and the control modifications another. Individual operations are used as building blocks of bigger ones. The aim is to define an operation that captures all the changes occurring to the grid on a single CAE step. This top operation is executed every time an action is issued. Thus the system starts at _grid i_ then an action is received and the top operation using as inputs the grid and action computes the next _grid i+1_, then the process is repeated.

To get a complete RL task from the previous description, at each step a reward and termination signal (if any) must be given. Thus a reward and a termination functions are defined usually depending on the grid. To complete the CAE, everything is packed into a single class with the gym API.

Consequently the architecture has three layers:

1. Primitive Operations
2. Top Operation
3. CA Environment
