.PHONY: help
help:
	cat Makefile


.PHONY: install
install:
	pip install -e .


.PHONY: install-develop
install-develop:
	@printf "Probably run: sudo make install-develop\n"
	npm i -g gitmoji-cli
	npm install git-br -g


.PHONY: hooks
hooks:
	gitmoji -i


.PHONY: build
build:
	python -m build


.PHONY: git-config
git-config:
	# `git br` shows branchs descriptions `git root` prints the project root
	git config --global alias.br !git-br # git branch --edit-description
	git config --global alias.br-describe 'branch --edit-description'
	git config --global alias.root 'rev-parse --show-toplevel'
	git config push.autoSetupRemote true


.PHONY: conda_env
conda_env:
	mamba env create --file "environment.yaml"


.PHONY: conda_env_rm
conda_env_rm:
	mamba remove --name gymca --all


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


.PHONY: clean
clean:
	find ./ -type d -name '__pycache__'        | xargs -I{} rm -r {}
	find ./ -type d -name '*.egg-info'         | xargs -I{} rm -r {}
	find ./ -type f -name '*~'                 | xargs -I{} rm    {}
	find ./ -type f -name 'monkeytype.sqlite3' | xargs -I{} rm    {}
	find ./ -type d -name '.pytest_cache'      | xargs -I{} rm -r {}
	find ./ -type d -name '.mypy_cache'        | xargs -I{} rm -r {}
	find ./ -type f -name 'TMP*'               | xargs -I{} rm    {}
	@git clean -d -n # To remove them change -n to -f
	@printf "\n\nTo remove git untracked files run:\ngit clean -d -f\n"


.PHONY: trailing-rm
trailing-rm:
	find gym_cellular_automata -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;


.PHONY: count
count:
	# Counts lines of code
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | perl -ne 'if(not /^ *?#/){print $$_}' | wc -l


.PHONY: gifs
gifs:
	# Create gifs for the environments registered at gymca.envs
	./scripts/gifs
