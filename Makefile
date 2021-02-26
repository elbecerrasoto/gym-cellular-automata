help:
	cat Makefile

install:
	pip install -e .

style:
	black .

test:
	pytest

test_forest_fire: 
	pytest gym_cellular_automata/tests/envs/forest_fire/
