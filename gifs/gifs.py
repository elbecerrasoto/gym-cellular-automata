from pathlib import Path
from typing import Union

import gif
from gym import Env, make

import gym_cellular_automata as gymca

gif.options.matplotlib["dpi"] = 200


def get_make_frame(env: Env):
    env.reset()

    @gif.frame
    def make_frame():
        env.render()
        _, _, done, _ = env.step(env.action_space.sample())

        if done:
            env.reset()

    return make_frame


def generate_gif(
    env: Env,
    name: Union[str, Path],
    each: int = 1,
    updates: int = 50,
    duration_between: int = 50,
):
    make_frame = get_make_frame(env)
    frames = []

    for i in range(updates):
        if i % each == 0:
            frames.append(make_frame())

    if isinstance(name, str):
        folder = Path().cwd() / 'gifs' 
        folder.mkdir(exist_ok=True)
        path = folder / f"{name}.gif"

    gif.save(frames, str(path), duration=duration_between)


if __name__ == "__main__":
    env = make("ForestFireHelicopter5x5-v1")
    generate_gif(env, "forest_fire_0", updates=20, duration_between=100)
