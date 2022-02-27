# Forest Fire Environment Bulldozer V1 #

![Forest Fire Helicopter](./../../../pics/render_bulldozer.svg)

## Description ##

This environment simulates a wild fire. The task is to extinguish the fire by controlling a _bulldozer_ that removes trees to prevent its advance.

The underlying wild fire simulation is a Cellular Automaton (CA) with rules similar to those of Drossel and Schwabl (1992).

At each time step the _bulldozer_ is at position $p_t$ and moves to a new position $p_{t+1}$, within a Moore's neighbourhood of its current position $p_t$.

After moving the _bulldozer_ can cut down a tree to prevent further fire spread into the current cell.

The reward is maximized when the forest is kept with its maximum number of tree cells, so the bulldozer must judiciously remove trees.

The cell states are the following:
1. *empty*
2. *burned*
3. *tree*

A _tree_ cell changes to *empty* when removed by the *bulldozer* or consumed by fire.

An internal clock is maintained so after some *bulldozer* actions the fire keeps spreading into its surroundings.

The simulation ends when there is no more fire, either by some success of the *bulldozer* or by a complete destruction of the forest.

## Cellular Automaton ##

## Observations ##

## Actions ##

## Reward ##
