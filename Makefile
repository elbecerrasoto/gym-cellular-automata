help :
	cat Makefile

install :
	pip install -e .

conda_env :
	conda env create --file "environment.yaml"

develop :
	pre-commit install
	sudo npm i -g gitmoji-cli
	gitmoji -i
	pip install -e .

hooks :
	pre-commit install
	gitmoji -i

style :
	isort ./
	black ./

test :
	mypy --config-file mypy.ini ./
	pytest ./

patch :
	./scripts/versionate -v --do "patch_up"

clean :
	find ./ -type d -name "__pycache__" | xargs rm -rf
	find ./ -type d -name "*.egg-info" | xargs rm -rf

count :
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | wc -l

.PHONY : help install conda_env develop hooks style test patch clean count
