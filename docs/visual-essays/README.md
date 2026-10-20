# Visual essay reference library

A good visual essay is an argument you can operate. The reader changes something, and the change is the evidence. The pages collected here do that in different ways: a slider, a draggable point, a playable game, a scroll-linked diagram, a shader you can edit.

This folder is a working reference shelf, not a hall of fame. Every entry names the mechanism worth stealing and how it maps onto marimo.

Open [index.html](index.html) for the browsable gallery. If you are serving the folder, run:

```bash
python3 -m http.server 8000 --directory docs/visual-essays
```

Then open <http://localhost:8000/>. The gallery also opens from the filesystem; the Markdown notes do not render there. The catalog lives in [references.json](references.json), the screenshots in [assets/](assets/), and the capture tool in [capture.py](capture.py). The interaction recreations are in [recreations/](recreations/).

## Start here

| If you read six | Why |
| --- | --- |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | Two sliders and one trajectory. The best compact example of direct manipulation carrying an argument. |
| [Gears](https://ciechanow.ski/gears/) | A physical mechanism you can take apart. The gold standard for spatial explanation. |
| [Principal Component Analysis](https://setosa.io/ev/principal-component-analysis/) | One draggable cloud, one derived quantity. A small idea executed completely. |
| [Build Your Own Database](https://www.nan.fyi/database) | A system built step by step, with a persistent artifact that changes under the reader. |
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | A single geometric picture that carries a research thesis. |
| [Parable of the Polygons](https://ncase.me/polygons/) | The mechanic is the moral argument. Nothing is explained that the reader has not just done. |

## The patterns

### Direct manipulation

The reader acts on the thing itself, not on a form that describes it.

- [Setosa: PCA](https://setosa.io/ev/principal-component-analysis/) drags the data points.
- [Setosa: Markov chains](https://setosa.io/ev/markov-chains/) links a state diagram to a transition matrix.
- [Distill: momentum](https://distill.pub/2017/momentum/) puts the step size and momentum next to the trajectory.
- [Immersive Math](https://immersivemath.com/ila/index.html) pairs every equation with a figure you can move.

What to steal: one handle, one derived quantity, one immediate redraw. Put the numeric readout next to the visual. If a slider is standing in for a spatial action, use the spatial action instead.

### Progressive construction

The same diagram returns, gaining one element at a time.

- [A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/) grows the same graph from nodes to messages to pooling.
- [R2D3: A Visual Introduction to Machine Learning](https://r2d3.us/visual-intro-to-machine-learning-part-1/) adds one decision boundary per scroll step.
- [Build Your Own Database](https://www.nan.fyi/database) adds one mechanism to the same file, then compacts it.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) isolates one matrix operation per figure and keeps token colors stable.

What to steal: a fixed layout and one new element per step. The reader should never rebuild the mental image. If the diagram has to move, animate the move and keep the old state visible.

### Playable simulation

The reader sets a rule, runs it, and sees the aggregate.

- [Parable of the Polygons](https://ncase.me/polygons/) turns a segregation preference into a board.
- [The Evolution of Trust](https://ncase.me/trust/) turns a payoff matrix into a repeated game.
- [Explorable Explanations](https://explorabl.es/all/) is the index for the genre.

What to steal: a small rule with a visible consequence. Let the reader break it, reset it, and compare two runs. Do not hide the initial condition.

### Annotated visual grammar

The pictures share a small vocabulary, and the vocabulary is the teaching.

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) uses lines, boxes, and circles until the reader can read a network.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) uses residual streams, heads, and paths as a grammar.
- [Feature Visualization](https://distill.pub/2017/feature-visualization/) makes a visual atlas of model features.
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) lays out attribution graphs left to right.

What to steal: define the primitives once, then reuse them. Annotate the one element that carries the point. A diagram that changes its visual language every section is a slideshow, not an argument.

### Long-form structure

Not every good essay is interactive. The best text-first essays win on hierarchy and evidence.

- [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) organizes a literature as a family tree and keeps notation stable.
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) uses one canonical system diagram as the section map.
- [Ink & Switch: Local-first software](https://www.inkandswitch.com/local-first/) pairs each principle with a prototype or a concrete scenario.
- [Distill: Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) links several views into one interface.

What to steal: a thesis, a small number of principles, and one piece of evidence per principle. End with the cases where the thesis fails.

### Live code, shaders, and 3D

Use these when the medium is the argument, not as decoration.

- [The Book of Shaders](https://thebookofshaders.com/) teaches GLSL through editable shaders.
- [Maxime Heckel](https://blog.maximeheckel.com/posts/beautiful-and-mind-bending-effects-with-webgl-render-targets/) art-directs the shader and the page together.
- [Bartosz Ciechanowski: Cameras and Lenses](https://ciechanow.ski/cameras-and-lenses/) is the benchmark for a 3D explainer.
- [Three.js examples](https://threejs.org/examples/) and [WebGPU samples](https://webgpu.github.io/webgpu-samples/) are parts bins, not essays.

What to steal: one primitive at a time. Start from the smallest triangle or compute pass. Keep a fallback for browsers without the feature. A 3D scene needs a spatial claim; otherwise a good 2D diagram is clearer.

## The marimo translation

| Pattern | Marimo primitive | Notes |
| --- | --- | --- |
| Direct manipulation | `mo.ui.anywidget` with a canvas | Keep the pointer state in JS and sync only the selected value. A slider is a weaker substitute. |
| Parameter sweep | `mo.ui.slider` plus Altair or anywidget | Redraw the whole figure on change. Do not animate if the comparison is the point. |
| Scrollytelling | CSS `position: sticky` around `mo.Html` | Marimo has no scroll progress hook. Use a step radio or slider if the scroll coupling is fragile. |
| Progressive construction | One `mo.Html` SVG function with a `step` argument | Keep the layout fixed. Add one element per step. |
| Linked views | Shared Python state, `mo.ui.altair_chart` selection | A click in one chart should be visible in every other view without a legend. |
| Playable simulation | Python state plus `mo.ui.run_button` | Keep the initial condition visible. Provide reset and pause. |
| Live code | `mo.ui.code_editor` plus anywidget | Run on demand, not on every keystroke. |
| 3D | Anywidget with Three.js or WebGPU | Use it for space, not for a static chart. Always provide a 2D fallback. |

The self-modeling essay already uses the dial (direct manipulation), the predict/reveal (a small commitment), the annotated MNIST chart (visual grammar), and the score matrix (a table as argument). The missing patterns are progressive construction and linked views.

## What not to copy

- Interaction without a claim. If the reader can change a parameter and the takeaway does not change, cut the control.
- 3D as decoration. A rotating object that could be a labeled diagram is a tax on the reader.
- Scroll hijacking. Sticky figures are useful; locking the scroll is not.
- Tiny controls. If the slider is 40px wide on a phone, the interaction does not exist.
- Animation that cannot pause. Add play, pause, and reduced motion.
- Hidden state. Show the current value, the initial value, and the comparison.
- A gallery of unrelated chart types. One visual grammar per essay.

## Capture the references

The capture tool uses an already-running Chrome DevTools endpoint. It opens each selected URL in a tab, waits for the page to settle, scrolls to an optional target, records a few interaction counts, and saves a JPEG.

Start Chrome with a debugging port:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless=new --remote-debugging-port=9222 about:blank
```

Capture everything with `"capture": true`:

```bash
uv run --script docs/visual-essays/capture.py
```

Capture one entry, or force a recapture:

```bash
uv run --script docs/visual-essays/capture.py --only distill-momentum --force
```

The catalog supports a `scroll_y` or a CSS `selector` per entry, which is how the tool reaches a figure below the hero. The output is `assets/<slug>.jpg` plus `captures.json`. The gallery is generated from the catalog and the captures:

```bash
uv run --script docs/visual-essays/build_gallery.py
```

## Recreations

The [recreations](recreations/) folder contains small, dependency-light pages for the patterns that are awkward to express with a chart:

- [direct manipulation](recreations/direct-manipulation.html): a Canvas 2D cloud with a compression slider and a stop-gradient toggle.
- [GLSL field](recreations/glsl-field.html): a WebGL2 fragment shader with a frequency control.
- [Three.js scene](recreations/three-scene.html): a 3D object with orbit controls and a distortion slider.
- [WebGPU field](recreations/webgpu-field.html): a WGSL full-screen shader with a fallback if WebGPU is unavailable.

Each page is a scaffold, not a finished essay. The point is to have a verified starting point for the next interaction.

## Credits

The catalog was assembled from the public pages linked above. Screenshots are for reference and critique. The pages remain the work of their authors.

If a page blocks the capture tool, add a `scroll_y` or `selector`, or capture it manually and drop the image into `assets/`.
