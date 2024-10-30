# Contributing

Steps

1. Set-Up
2. Code
3. Integrate

## Set-Up

+ Download

```bash
  git clone https://github.com/elbecerrasoto/gym-cellular-automata
```

+ Set-up the development environment
  + Requires [anaconda](https://www.anaconda.com/) for managing the virtual environment
  + I recommend using the following distribution [mini-forge](https://github.com/conda-forge/miniforge/#download)

```bash
  cd gym-cellular-automata
  mamba env create --file "environment.yaml"
  mamba activate gymca
```

+ Install gym-cellular-automata

```bash
pip install -e .
```

+ Check that everything is fine

```bash
pytest -m "not slow" --maxfail=3 ./gym_cellular_automata
```

+ :octocat: Checkout a new development branch
```
git checkout -b awesome-feat
```


## Code

+ :space_invader: Code

+ Now and then format and test your code
``` bash
    make style
    make test
```

## Integrate

+ Open a _pull request_
+ Wait for your changes to be accepted
+ Iterate all over again

## Guidelines

+ Use type annotations if possible
+ Format the code
  + `make style`
+ Do not break the build, check it by running the test suite
  + `make test`

+ Optionally, use gitmoji, requires `sudo`
```bash
sudo npm i -g gitmoji-cli
gitmoji -i # make hooks
```
