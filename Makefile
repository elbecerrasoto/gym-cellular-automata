.PHONY: help
help:
	cat Makefile


.PHONY: install
install:
	pip install -e .


.PHONY: install-develop
install-develop:
	# sudo make install-develop
	npm i -g gitmoji-cli
	npm install git-br -g


.PHONY: build
build:
	python -m build


.PHONY: git-aliases
git-aliases:
	# `git br` shows branchs descriptions `git root` prints the project root
	git config --global alias.br !git-br # git branch --edit-description
	git config --global alias.br-describe 'branch --edit-description'
	git config --global alias.root 'rev-parse --show-toplevel'


.PHONY: conda_env
conda_env:
	conda env create --file "environment.yaml"


.PHONY: conda_env_rm
conda_env_rm:
	conda remove --name gymca --all


.PHONY: hooks
hooks:
	pre-commit install
	gitmoji -i


.PHONY: hooks-dry
hooks-dry:
	pre-commit run --all-files


.PHONY: hooks-update
hooks-update:
	pre-commit autoupdate


.PHONY: stye
style:
	isort ./
	black ./


.PHONY: test
test:
	pytest -m "not slow" --maxfail=3 ./gym_cellular_automata


.PHONY: test-debug
test-debug:
	# Depends on pip install pytest-ipdb
	pytest -m "not slow" --ipdb ./gym_cellular_automata


.PHONY: test-coverage
test-coverage:
	pytest -m "not slow" -x --cov=./gym_cellular_automata ./gym_cellular_automata


.PHONY: test-slow
test-slow:
	time pytest -m "slow" -x ./gym_cellular_automata


.PHONY: test-visual
test-visual:
	./scripts/update_gallery "helicopter" "bulldozer" --interactive -v --steps "128" "512"


.PHONY: linter
linter:
	# Finds debugging prints
	find ./gym_cellular_automata/ -type f -name "*.py" | sed '/test/ d' | xargs egrep -n 'print\(|ic\(' | cat
	mypy --config-file mypy.ini ./


.PHONY: patch
patch:
	./scripts/versionate -v --do "patch_up"


.PHONY: gallery
gallery:
	# Depends on GNU parallel
	time yes 1624 | head -12 | parallel ./scripts/update_gallery "bulldozer" -v --steps {}
	time yes 0 | head -6 | parallel ./scripts/update_gallery "helicopter" -v --steps {}


.PHONY: clean
clean:
	# Depends on trash-cli  https://github.com/andreafrancia/trash-cli
	find ./ -type d -name '__pycache__' | xargs -I{} trash {}
	find ./ -type d -name '*.egg-info' | xargs -I{} trash {}
	find ./ -type f -name '*~' | xargs -I{} trash {}
	find ./ -type f -name 'monkeytype.sqlite3' | xargs -I{} trash {}
	find ./ -type d -name '.pytest_cache' | xargs -I{} trash {}
	find ./ -type d -name '.mypy_cache' | xargs -I{} trash {}
	find ./ -name 'TMP*' | xargs -I{} trash {}
	git clean -d -n # To remove them change -n to -f
	echo "\n\nTo remove git untracked files run:\ngit clean -d -f"


.PHONY: trailing-spaces
trailing-spaces:
	find gym_cellular_automata -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;


.PHONY: count
count:
	# Counts lines of code
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | perl -ne 'if(not /^ *?#/){print $$_}' | wc -l


.PHONY: generate_gifs
generate_gifs:
	# Create gifs for the environments registered at gymca.envs
	./scripts/gifs
