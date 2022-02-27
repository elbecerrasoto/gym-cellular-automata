help :
	cat Makefile

install :
	pip install -e .

install-develop : # sudo make install-develop
	npm i -g gitmoji-cli
	npm install git-br -g

build :
	python -m build

git-aliases : # `git br` shows branchs descriptions `git root` prints the project root
	git config --global alias.br !git-br # git branch --edit-description
	git config --global alias.br-describe 'branch --edit-description'
	git config --global alias.root 'rev-parse --show-toplevel'

conda_env :
	conda env create --file "environment.yaml"

conda_env_rm :
	conda remove --name gymca --all

hooks :
	pre-commit install
	gitmoji -i

hooks-dry :
	pre-commit run --all-files

hooks-update :
	pre-commit autoupdate

style :
	isort ./
	black ./

test :
	pytest -m "not slow" --maxfail=3 ./gym_cellular_automata

test-debug : # Depends on pip install pytest-ipdb
	pytest -m "not slow" --ipdb ./gym_cellular_automata

test-coverage :
	pytest -m "not slow" -x --cov=./gym_cellular_automata ./gym_cellular_automata

test-slow :
	time pytest -m "slow" -x ./gym_cellular_automata

test-visual :
	./scripts/update_gallery "helicopter" "bulldozer" --interactive -v --steps "128" "512"

linter :
	# Finds debugging prints
	find ./gym_cellular_automata/ -type f -name "*.py" | sed '/test/ d' | xargs egrep -n 'print\(|ic\(' | cat
	mypy --config-file mypy.ini ./

patch :
	./scripts/versionate -v --do "patch_up"

gallery : # Depends on GNU parallel
	time yes 1624 | head -12 | parallel ./scripts/update_gallery "bulldozer" -v --steps {}
	time yes 0 | head -6 | parallel ./scripts/update_gallery "helicopter" -v --steps {}

clean : # Depends on trash-cli  https://github.com/andreafrancia/trash-cli
	find ./ -type d -name '__pycache__' | xargs -I{} trash {}
	find ./ -type d -name '*.egg-info' | xargs -I{} trash {}
	find ./ -type f -name '*~' | xargs -I{} trash {}
	find ./ -type f -name 'monkeytype.sqlite3' | xargs -I{} trash {}
	find ./ -type d -name '.pytest_cache' | xargs -I{} trash {}
	find ./ -type d -name '.mypy_cache' | xargs -I{} trash {}
	find ./ -name 'TMP*' | xargs -I{} trash {}
	git clean -d -n # To remove them change -n to -f
	echo "\n\nTo remove git untracked files run:\ngit clean -d -f"

trailing-spaces:
	find gym_cellular_automata -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

count :
	# Counts lines of code
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | perl -ne 'if(not /^ *?#/){print $$_}' | wc -l

.PHONY : help install install-develop build git-aliases conda_env conda_env_rm hooks hooks-dry hooks-update style test test-debug test-coverage test-slow test-visual linter patch gallery clean count
