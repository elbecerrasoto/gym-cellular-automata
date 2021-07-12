# Issues

## Emoji Font Warnings

The emoji font raises `RuntimeWarning`.
The font is used on the render method of the Bulldozer Environment.

The warning is raised when attempting to render the Figure.
On either hard-copy (e.g. `fig.savefig`) or interactive mode (e.g. `plt.show`).

Globally suppressing Warnings could be dangerous.
Thus the user would need to manually suppress the warning when visualizing the Figure object returned by `env.render()`.

A code snippet like this will do:
```python
def mute_warnings(callable_, *args, **kwargs):
    from warnings import catch_warnings, simplefilter

    with catch_warnings():
        simplefilter("ignore")
        callable_(*args, **kwargs)


fig = env.render()
mute_warnings(plt.show)
mute_warnings(fig.savefig, "bulldozer_render.svg")
```


**Warning message:**
```
./matplotlib/textpath.py:65: RuntimeWarning: Glyph 108 missing from current font.
  font.set_text(s, 0.0, flags=LOAD_NO_HINTING)

./matplotlib/textpath.py:65: RuntimeWarning: Glyph 112 missing from current font.
  font.set_text(s, 0.0, flags=LOAD_NO_HINTING)

./matplotlib/backends/backend_agg.py:241: RuntimeWarning: Glyph 108 missing from current font.
  font.set_text(s, 0.0, flags=flags)

./matplotlib/backends/backend_agg.py:241: RuntimeWarning: Glyph 112 missing from current font.
  font.set_text(s, 0.0, flags=flags)
```
