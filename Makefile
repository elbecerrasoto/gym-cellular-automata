dev_branch := dev

help:
	cat Makefile

install:
	pip install -e .

dev:
	git checkout $(dev_branch)
	git status

style:
	black .

test:
	pytest

test_forest_fire: 
	pytest gym_cellular_automata/tests/envs/forest_fire/
