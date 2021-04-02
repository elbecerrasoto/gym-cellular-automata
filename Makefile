help :
	cat Makefile

install :
	pip install -e .

develop : install
	pip install black
	pip install pytest
	pip install pytest-repeat
	pip install pytest-cov
	pip install pytest-randomly
	pip install pre-commit
	pre-commit install
	sudo npm i -g gitmoji-cli
	gitmoji -i

style :
	black ./

test :
	pytest ./

clean :
	find ./ -type d -name "__pycache__" | xargs rm -rf
	find ./ -type d -name "*.egg-info" | xargs rm -rf

.PHONY : help install develop style test clean
