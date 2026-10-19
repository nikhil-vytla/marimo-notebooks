# From paper to visual essay

Notes from building the self-modeling essay. The goal is not a notebook
that reproduces a paper. It is an argument a reader can operate.

## What earns an essay

A paper is a good candidate when most of these are true.

| test | question | red flag |
| --- | --- | --- |
| Surprise | Does the claim make a competent reader say "wait, really?" | The result is already intuitive |
| Mechanism | Can you draw the causal story, or turn a parameter that changes it? | The paper is a system description with no measurable claim |
| Data | Can you get the numbers, code, weights, or a cheap proxy? | Nothing released and no way to approximate |
| Lever | Is there an assumption the reader can change and see the consequence? | The only interaction would be a decorative slider |
| Gap | Is there a known disagreement, robustness question, or missing control? | Every panel is expected to reproduce cleanly |
| Budget | Can the hero figure run on hardware you have this week? | The first figure needs a multi-day GPU run |
| Audience | Can you name the person who will read it? | "Anyone interested in AI" |

A benchmark table with no mechanism is usually a bad essay. A negative
result with a clean control is usually a good one.

## The shape that works

The self-modeling essay follows this arc, and it generalizes.

1. Hook. One sentence, one surprising claim.
2. Puzzle. Make the reader commit to an answer before you show it.
3. Mechanism. The smallest model that produces the effect.
4. Hero evidence. One figure, paper values overlaid, n and intervals visible.
5. Generalization. Does it hold across sizes, datasets, or seeds?
6. Complication. A second dataset or a formal measure that disagrees.
7. Diagnosis. Probes, ablations, and controls.
8. Resolution. A scoreboard of what reproduced and what did not.
9. Open questions. What would change your mind.

The null results are not an appendix. In this essay they are the second
act, and they are why the piece is worth reading.

## Build order

### Stage 0: read for claims

Write a claim graph. Boxes are claims, arrows are "because" or "tested
by". Mark which claims have a figure behind them. A claim with no figure
is either an assumption or a place to look for a gap.

### Stage 1: figure inventory

Pick one hero figure, two supporting figures, and one negative result.
Read off three to five values from the paper and attach an error
estimate. Note where the caption and the text disagree. Those
disagreements are free material.

### Stage 2: build the oracle

Port the training code, write tests, and reproduce the hero figure for
one seed. Do not write essay prose yet. In this repo the oracle was the
reference trainer plus eleven tests. It caught the batched trainer; a
frozen-head control caught the gradient-clipping bug later.

### Stage 3: fix the data contract

Decide the JSON shape and the config before the sweep. Save every
per-run value, not just the mean. The essay should never train; it
reads files that a training notebook wrote.

### Stage 4: run the probes

Before believing a disagreement, run the ablations that could explain
it. Then put the probes in the essay even when they fail. A probe that
rules out your favorite explanation is more valuable than one more
seed of the same curve.

### Stage 5: write the narrative

Write the headings first. Fill them in after the data is stable. A
reader should be able to follow the argument with the code collapsed.

### Stage 6: add interaction

One control per concept. Add a prediction and a reveal. Remove any
control that only moves a parameter without changing what the reader
believes.

### Stage 7: verify and publish

`just check`, `just smoke`, refresh the saved session, then publish.
The session export matters: it lets molab show outputs without running
the notebook.

## The data contract

```text
notebooks/<topic>/
  README.md
  <topic>_essay.py          # reads data, never trains
  <topic>_train.py          # writes one JSON per panel
  public/data/<panel>.json
  __marimo__/session/<essay>.py.json
```

A panel file should carry its own provenance.

```json
{
  "config": {"hidden": 512, "batch": 512, "epochs": 50},
  "device": {"name": "RTX PRO 6000", "seconds": 43},
  "results": {"0": {"sd_mean": [], "sd_ci": []}},
  "runs": [{"seed": 0, "aw": 0, "sd": 0.0974}],
  "paper": {"0": 0.098},
  "provenance": {"commit": "4c634bc", "date": "2026-10-05"}
}
```

Rules that keep the essay honest:

- The essay never trains. It loads JSON and computes summaries.
- The config that produced the data travels with the data.
- Every aggregate keeps the per-run values.
- Every value read off a paper figure is labeled as approximate.
- A missing value is `null`, not zero.

## Interaction patterns

### Predict then reveal

A radio for the guess, a checkbox for the reveal, and a callout that
says whether the guess was right and why. The exercise near the end of
the self-modeling essay is this pattern. It works because the reader
has committed before seeing the number.

```python
guess = mo.ui.radio(
    options={"flip": "flip", "survive": "survive"},
    label="Your prediction",
)
reveal = mo.ui.checkbox(label="Reveal")
```

### Parameter sweep

A slider or dropdown, a chart that updates, and a fixed axis. If the
axis rescales on every change, the reader cannot compare two settings.
The AW slider on the loss section is this pattern.

### Linked views

A selection in one chart filters the table and the other chart. Vega-Lite
selections or a shared `mo.ui` control both work. Link only what the
reader would link in their head.

### Time machine

An epoch slider over saved snapshots. This is the animation primitive
you actually need; a video is harder to scrub. The weight histogram
slider is this pattern.

### Scale toy

A tiny analytic model that shows why a unit change moves a metric. The
activation-scale slider in the CIFAR section is this pattern. It takes
twenty lines and explains a whole panel.

### Ablation explorer

A dropdown of variants, the same metric and axis for each, and the
result table. The empty result is a result. The CIFAR probe table is
this pattern.

### Contrast two datasets

Run the same ablation on the dataset where the effect is real and the
one where it is not. The contrast is the argument. In the self-modeling
essay, dropping the stop-gradient reverses the MNIST effect and barely
moves CIFAR. That pair is more convincing than either result alone, and
it keeps the reader from overgeneralizing.

### What would change my mind

A short list, not a caveat paragraph. Make each item concrete enough
that someone could run it. A falsifier is more useful than a hedge.

### Code peek

An accordion with the relevant snippet. The reader who trusts you can
skip it; the reader who does not can check. Keep the snippet short
enough to read in one screen.

One control per concept. If two controls do the same job, delete one.
If an interaction does not change a belief, it is decoration.

## The marimo ladder

| level | what it buys | APIs | cost |
| --- | --- | --- | --- |
| 0 Static | A readable essay with LaTeX and charts | `mo.md`, `mo.Html`, `mo.image` | None |
| 1 Reactive | Sliders, dropdowns, linked charts, tables | `mo.ui.*`, `mo.ui.altair_chart`, `mo.ui.table` | Low |
| 2 Bespoke visuals | SVG diagrams, custom layout, typography | `mo.Html`, inline SVG, `mo.App(css_file=...)` | Medium |
| 3 Anywidget | Canvas, WebGL, drag, gamepad, audio | `mo.ui.anywidget`, `_esm`, traitlets | High, you own the JS |
| 4 WASM | Python in the browser, no server | `marimo export html-wasm`, `--sandbox` | Medium, package limits |
| 5 Series | A landing page and several linked notebooks | `mo.notebook_location`, static export, Pages | Medium |
| 6 Living essay | Re-run on new data, chat over the paper | `mo.ui.chat`, `marimo pair`, `mo.persistent_cache` | High, cost and provenance |
| 7 Publication | Static site, interactive islands, citation | `marimo export html`, hosting | Medium |

Most essays should stop at level 2 or 3. Level 3 is where a visual essay
starts to feel different from a notebook, but only if the interaction
carries the argument. A custom canvas that explains nothing is worse
than a good chart.

The frontier, in order of payoff:

1. In-browser training toys. A tiny model that trains in the reader's
   browser while they change a parameter. WASM makes this possible for
   small numpy or torch-free examples. The cost is package support and
   browser memory.
2. Direct manipulation of the mechanism. Dragging a vector, a token, or
   a decision boundary is more convincing than a slider. This needs
   anywidget, and it needs a fallback for keyboard users.
3. A living replication dashboard. One page that runs the training
   notebooks, stores new results, and diffs them against the paper.
   This is a multi-notebook project, not a cell.
4. A paper chat that cites the data. Useful for exploring a long paper,
   dangerous if it becomes the only way to read the essay. Keep the
   written argument primary.
5. A course. A sequence of essays that build on each other, with a
   shared vocabulary and exercises. This is level 5 and a different
   product from a single essay.

## Where marimo stops

- It is not a general web framework. No auth, multi-user state,
  payments, routing, or SEO. If you need those, build a web app and
  embed or link the notebook.
- WASM is not a GPU. Test the export early, because package
  availability is the constraint.
- Molab sessions last twelve hours and stop after ninety minutes idle.
  Checkpoint long runs and do not rely on local writes. Only files
  uploaded through the sidebar or written with `mo.persistent_cache`
  persist.
- Large outputs slow the browser. Downsample, precompute, and avoid
  rendering a full data frame when a summary will do.
- Animation is not a first-class primitive. Use a slider over
  snapshots, CSS transitions, or JS. Do not promise a video.
- Accessibility is on you. Custom HTML needs labels, keyboard
  handling, and a non-interactive fallback.
- `marimo check` catches structure, not numerical drift. Pin versions,
  seeds, and data hashes.

## Working with agents

- Give the agent a known-good oracle. Without one, an agent can produce
  a plausible notebook that is wrong in a way you will not notice.
- Give it an acceptance test it can run: `just check`, `just smoke`, a
  specific number, a screenshot.
- Give it a data contract. Agents fill a schema well and invent the
  wrong schema badly.
- Run independent tasks in parallel. Scaffold, training, essay, and
  reference copy can go at once.
- Review the numerical claims yourself. The agent can port code; you
  own the claim.
- Keep the training notebook out of the essay path. The essay should
  load in seconds.
- Use `marimo pair` for a live kernel. Do not edit the `.py` file while
  a session is running; the kernel is the source of truth.
- Keep a decision log and a corrections section. The corrections are
  often the most valuable part of the essay.

## The self-modeling essay: next moves

1. **CIFAR architecture gap.** The skip connection from block 3 into the
   hidden layer is the biggest unresolved difference. Implement it
   faithfully and rerun 3A and 3B. If the inversion flips, you have a
   result. If it does not, you have a stronger negative.
2. **More seeds for the slow-head control.** Three seeds and 100 epochs
   is a sign. Ten seeds and 250 epochs is a measurement. Pre-register
   the read-out before running.
3. **Rank diagnostic for IMDB.** The hidden layer is 128 units from a
   64-dimensional embedding. Compute the rank and singular values of
   the self-model rows and the hidden activations. Test whether the
   self-model learns a projection, as the weight distance suggests.
4. **Scale-invariant read-out.** Weight SD is scale-dependent; RLCT is
   the paper's answer, but your RLCT is flat. Add a normalized SD
   (SD times hidden norm) to show whether the CIFAR widening is only
   units.
5. **Essay polish.** Add a guided path for sections 1 to 4 and a lab
   path for the rest. Add a sidebar table of contents. Add a download
   button for the data. Check the essay on a phone.
6. **Split if it grows.** The essay is about 2,800 lines. The next
   addition should probably be a second notebook, not another section.
   Keep the essay under a twenty-minute read.

## A paper queue

Use one row per paper. Replace the examples with your own list.

| paper | claim | hero figure | data | compute | interaction | why now | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| example: lottery ticket | sparse subnetworks train as well as the dense net | accuracy vs sparsity | released masks | one GPU-day | drag a sparsity mask | cheap, visual | queued |
| example: grokking | generalization arrives long after memorization | train/test loss vs steps | small | CPU | step through training | phase transition | queued |

Good categories for the next few:

- Mechanistic interpretability with a circuit you can ablate. Toggle a
  head or feature and watch the logits. The interaction is the
  argument.
- Scaling laws or phase transitions. A log-log plot with a residual
  panel and a slider for the fit window.
- Diffusion or flow models. Step through denoising and change the
  guidance. This is the most naturally visual family in ML.
- Causal inference. Change the DAG and watch the estimate. The reader
  learns by doing the confounding.
- RL or agents. Let the reader play the environment or change the
  reward. This needs a small environment and a clear scoring rule.
- Bayesian or decision papers. Change the prior and watch the posterior.
  The Bayes notebook is the warm-up.
- Numerical optimization. Change the step size or conditioning and watch
  convergence. Cheap to run, easy to make interactive.

Avoid papers whose main contribution is a large benchmark table with no
mechanism, no released data, and no cheap proxy. They make good blog
posts, not good essays.

## The quality bar

- The claim is one sentence and appears on the first screen.
- Every figure has n, an interval or a reason why not, and a provenance
  line.
- At least one prediction and reveal.
- At least one control the reader can change and see a consequence.
- At least one negative result or unresolved question.
- The essay loads in under five seconds and no cell takes more than ten
  seconds.
- `just check` and `just smoke` pass.
- The saved session is current, so molab shows outputs without running.
- A reader can get from the essay to the training notebook and the raw
  data.
- The text says what would change your mind.

## First ninety minutes on a new paper

- Twenty minutes: read for claims, not sentences. Write the claim graph.
- Twenty minutes: find the data or code and the hero figure. Read off
  three to five numbers.
- Twenty minutes: run the cheapest version of the hero figure. One seed
  is enough.
- Thirty minutes: sketch the headings and the one interaction that
  carries the argument.

If the hero figure does not run in twenty minutes, the paper is a bigger
project than a first pass. Say so early.

## Things I would do differently

- Build the oracle and tests before the first essay cell. The
  self-modeling reproduction had that, and it saved the later work.
- Write the corrections section in week one. Batch size, preprocessing,
  and detach were found late; they are the most useful part now.
- Put the scoreboard in the middle, not the end. It tells the reader
  what to trust before the complications.
- Treat a null result as a first-class finding. The CIFAR and IMDB
  sections are why the essay is interesting.
- Run every key ablation on both the positive and negative dataset. The
  stop-gradient looked like a detail until it reversed MNIST and did
  nothing to CIFAR.
- Keep the essay and training notebooks separate from day one.
- Use `just` and CI before the second notebook. It makes agent work
  reviewable.
- Do not polish prose until the data contract is stable. Every data
  change otherwise breaks the narrative.
