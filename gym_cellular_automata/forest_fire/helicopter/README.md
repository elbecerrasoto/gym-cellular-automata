# Forest Fire Environment Helicopter V0 #

![Forest Fire Helicopter](./../../../pics/render_helicopter.svg)

## Description ##

The task is to put down a wild fire by controlling a helicopter.

The fire is simulated by a Forest Fire Automaton [Drossel and Schwabl (1992)].

At each time step the helicopter moves to an adjacent cell and if it is a *fire* it changes it to a *empty* cell.

The reward is $+1$ per tree cell and $-1$ per *fire* cell.

The system has no termination state.

The actions are the natural numbers from 0 to 8, each representing a direction:
0. Left-Up
1. Up
2. Right-Up
3. Right
4. Don't move
5. Left
6. Left-Down
7. Down
8. Right-Down

## Cellular Automaton ##

Forest Fire Automaton Drossel and Schwabl (1992)

Three type of cells: TREE, EMPTY and FIRE
At each time step and for each cell apply the following rules (order does not matter).
* **Lighting Rule** With probability f:
	+ TREE turns into FIRE
* **Propagation Rule** If at least one neighbour is FIRE:
	+ TREE turns into FIRE
* **Burning Rule** Unconditional:
	+ FIRE turns into EMPTY
* **Growth Rule** With probability p:
	+ EMPTY turns into TREE
