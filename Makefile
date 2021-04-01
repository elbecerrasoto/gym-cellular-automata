help:
	cat Makefile

install:
	pip install -e .

develop:
	pip install black
	pip install pytest
	pip install pytest-repeat
	pip install pytest-cov
	pip install pytest-randomly

style:
	black .

test:
	pytest

clean:
	find ./ -type d -name "__pycache__" | xargs rm -rf
	find ./ -type d -name "*.egg-info" | xargs rm -rf
