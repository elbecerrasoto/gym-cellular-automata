help:
	cat Makefile

install:
	pip install -e .

style:
	black .

test:
	pytest

clean:
	find ./ -type d -name "__pycache__" | xargs rm -rf
	find ./ -type d -name "*.egg-info" | xargs rm -rf
	find ./ -type f -name "ipython.html" | xargs rm -f
