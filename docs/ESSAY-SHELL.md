# The essay shell

`templates/essay.py` is a starting point for a visual essay, not a
notebook. It gives you the frame, the typography, and four reusable
patterns. Replace the placeholders and keep the parts that help the
argument.

## Start one

```bash
just new-essay my-paper
```

That creates `notebooks/my-paper/` with the essay and an empty
`public/data/`. Open it with `just edit notebooks/my-paper/my_paper.py`.

## What to replace

- `META`: title, lede, paper, authors, status, reading time, update date,
  repository.
- `SCORE`: one row per paper panel. Use `yes`, `partly`, or `no`.
- The section prose. Keep the headings if they fit the argument, or
  change them. The shell is a starting point, not a contract.
- The predict/reveal question and the result behind it.
- The provenance footer.

## The layout

```text
notebooks/<slug>/
  README.md
  <snake_slug>.py        # the essay
  public/data/           # the data the essay reads
  __marimo__/session/    # saved outputs for molab
```

The essay reads data. It never trains. The training notebook writes
JSON into `public/data/`, and the essay loads it with
`mo.notebook_location() / "public" / "data" / <file>`.

## The four patterns

### The hero

The hero is an HTML block with a kicker, a title, a lede, and a status
pill. It should make the claim in the first screen. Keep the title
short enough to read on a phone.

### The scoreboard

`SCORE` is a list of dicts. Each row becomes a card with a panel name,
a status, and one sentence of evidence. Use it before the
complications, not after. The reader should know what to trust before
they read the failures.

### Predict then reveal

The `guess` and `reveal` cells are the pattern. Ask a question the
reader can answer before seeing the data. The reveal should say whether
the guess was right and include the number that settles it. Do not
reveal automatically; the commitment is the point.

### The provenance footer

The footer names the paper, the data, and the method. It should link to
the repository and to the training notebook. A reader who wants to
check you should not have to guess where the numbers came from.

## Typography and CSS

The shell injects its CSS in the first cell with `mo.Html`. Nothing is
loaded from the network, so the style travels with the notebook to
molab and to an HTML export.

The palette is four colors: ink, paper, amber accent, and a muted gray.
Status colors are green for reproduced, amber for partly, and red for
not reproduced. Keep the palette small. A fifth color should mean
something.

If you need a custom font or a heavier layout change, put a CSS file
next to the notebook and use `marimo.App(css_file="essay.css")`
instead. The inline style is the portable default.

## Reading mode

The shell marks boilerplate cells with `hide_code=True`. For a
published read, run the essay with `marimo run` or export it to HTML.
In edit mode the code is still one click away, which is useful while
you are building.

## Checks

The template is part of `just check` now. Before you publish an essay:

```bash
just check
just smoke notebooks/<slug>/<slug>.py
just session notebooks/<slug>/<slug>.py
```

Refresh the saved session after every change. Molab uses it to show
outputs without running the notebook.

## Direct manipulation

If the essay needs a hero interaction, the shell stops short on purpose.
The compression dial in the self-modeling essay is the reference: an
`anywidget` that drags, toggles, and interpolates precomputed frames.
Copy the shape, not the file. Keep the widget's data in
`public/data/`, keep the Python readout in a separate cell, and test the
keyboard path as well as the pointer path.

## When not to use the shell

If the essay is one figure and three paragraphs, a plain notebook is
fine. If the essay needs a multi-page site, authentication, or a
database, build a web app and embed the notebook. The shell is for
essays that are a single argument with a small number of interactions.

## See also

- [docs/ESSAY-PLAYBOOK.md](ESSAY-PLAYBOOK.md) for choosing papers and
  the interaction patterns.
- [templates/essay.py](../templates/essay.py) for the source.
- [self_modeling_guided.py](../notebooks/self-modeling/self_modeling_guided.py)
  for the shell filled in with a real essay.
