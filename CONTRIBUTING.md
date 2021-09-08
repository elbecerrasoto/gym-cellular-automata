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

```bash
  cd gym-cellular-automata
  make conda_env
  conda activate gymca
```

+ Install

```bash
make install
```

+ Create hooks for `git`
  + This requires [gitmoji](https://github.com/carloscuesta/gitmoji)
  + It can be installed via `npm` with the following command
  + `npm i -g gitmoji-cli`

```bash
make hooks
```

+ Check that everything is fine

```bash
  make test
```

+ :octocat: Checkout a new development branch
```
  checkout -b awesome-feat
```

### Code

+ :space_invader: Code

+ Across coding sessions, pull and merge to get the latest hotness from _main_
```bash
  git switch main
  git pull
  git checkout awesome-feat
  git merge main
```

+ Now and then format and test your code
``` bash
    make style
    make test
```

### Integrate

+ Open a _pull request_
+ Wait for your changes to be accepted
+ Iterate all over again

## Guidelines

+ Use type annotations if possible
+ Format the code
  + `make style`
+ Do not break the build, check it by running the test suite
  + `make test`
+ Test your code
