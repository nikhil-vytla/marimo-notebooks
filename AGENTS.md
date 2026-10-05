# AGENTS.md

Conventions for this repo of self-contained marimo notebooks. Read this before adding or changing a notebook.

## Layout

```text
notebooks/<kebab-slug>/
  README.md              # title, status, molab link, sources, file list
  <snake_slug>.py        # entry notebook, self-contained
  public/                # data and assets the notebook reads
templates/notebook.py    # start every new notebook here
justfile                 # check, smoke, edit, new, list
```

One topic per folder under `notebooks/`. Name the entry notebook after its folder (`bayes-theorem/` -> `bayes_theorem.py`); add more descriptively named notebooks when a topic needs several.

## Add a notebook

1. Run `just new <kebab-slug>` to scaffold `notebooks/<slug>/` from `templates/notebook.py`.
2. Add a PEP 723 `# /// script` header listing every import, including `marimo`, with exact `==` versions.
3. Write the notebook source.
4. Fill in the folder `README.md` and add a root `README.md` table row.

Each notebook imports only from itself. Write the code directly in the notebook.

## Edit and run a notebook

- Open for editing: `just edit notebooks/<slug>/<snake_slug>.py` (runs `uvx marimo edit --sandbox <file>`).
- Run headless in script mode: `just smoke notebooks/<slug>/<snake_slug>.py` (sets `SMOKE=1`). Read `SMOKE` to shrink work so this finishes in under a minute.
- Training or GPU notebooks put `# smoke: skip` in their first 30 lines; CI then skips them.
- Save outputs for essay-style notebooks: `just session notebooks/<slug>/<snake_slug>.py` commits state under `__marimo__/session/` so molab shows outputs without running. Re-run it whenever the essay or its data changes. Never do this for training notebooks.
- Running non-interactively: close stdin (`< /dev/null`); `marimo --sandbox` can wait for input.
- Live pairing: when a marimo session is open on a notebook, edit through the kernel (`marimo-pair` skill / `marimo._code_mode`) instead of writing the `.py` file.

## Data and assets

- Put data and assets in `public/` beside the notebook.
- Load them with `mo.notebook_location() / "public" / <path>`.
- This resolves locally, on molab, and in WASM exports.

## Notebook structure

- Keep expensive work behind `mo.ui.run_button`; opening a notebook stays cheap.
- Essay notebooks read precomputed data; put training in its own notebook.
- Use `_`-prefixed names for cell-local variables.

## Definition of done

A notebook change is done when all hold:

- `just check` passes (`marimo check --strict`, `scripts/check_empty_cells.py`, `typos`, and ruff; warnings fail).
- `just smoke <path>` runs in script mode.
- The folder `README.md` and the root `README.md` table are updated.

## Gotchas

- marimo: one cell owns each public name; `_`-prefixed names stay cell-local; no wildcard imports.
- PEP 723: install dependencies with `--sandbox` (`just edit` does this).
- molab: toggle GPU per notebook; sessions run at most 12h; shutdown after 90 min idle; only files uploaded through the sidebar or written with `mo.persistent_cache` persist.
- molab link: `https://molab.marimo.io/github/<owner>/<repo>/blob/main/<path>`. It resolves once the file is on `main`; for a branch, replace `main` with the branch name.
