help :
	cat Makefile

install :
	pip install -e .

install-develop : # sudo make install-develop
	npm i -g gitmoji-cli
	npm install git-br -g

develop :
	git config --global alias.br !git-br

conda_env :
	conda env create --file "environment.yaml"

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

test-debug :
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

gallery :
	./scripts/update_gallery "helicopter" "bulldozer" -v --out "./pics/tmp_helicopter.svg" "./pics/tmp_bulldozer.svg" --steps "64" "1066"

clean : # Depends on trash-cli  https://github.com/andreafrancia/trash-cli
	find ./ -type d -name "__pycache__" | xargs -I{} trash {}
	find ./ -type d -name '*.egg-info' | xargs -I{} trash {}
	find ./ -type f -name '*~' | xargs -I{} trash {}
	find ./ -type f -name "monkeytype.sqlite3" | xargs -I{} trash {}
	trash ./pics/tmp_bulldozer.svg  ./pics/tmp_helicopter.svg
	git clean -d -n # To remove them change -n to -f
	echo "\n\nTo remove git untracked files run:\ngit clean -d -f"

count :
	# Counts lines of code
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | perl -ne 'if(not /^ *?#/){print $$_}' | wc -l

.PHONY : help install install-develop develop conda_env hooks hooks-dry hooks-update style test test-debug test-coverage test-slow test-visual linter patch gallery clean count
