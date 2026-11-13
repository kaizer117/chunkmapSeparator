# AGENTS.md — Chunk Map Separator

This document is for an agent (or a human) about to change this repository. It
describes what the code actually does, which behaviours are load-bearing, and
which parts only look like dead weight. **Read it before refactoring.** The
short version: this is not a generic SVG exporter. It is a small cartographic
pipeline with two different vectorisation contracts, a deliberately biased set
of accuracy metrics, and a random island generator whose label map is part of
the vectorisation API.

The README is the user manual. The research notes live under `research/`. This
file is the maintenance/refactor contract.

---

## 0. Repository layout and import rules

The repository was reorganized so that every kind of file has one home. Keep it
that way; do not recreate a flat pile of modules.

```
main.py                    root entry point; the only command users run
chunkmap/                  all operational code (importable package)
  cli.py                   argument parser + dispatch
  accuracy.py              measurement and the standalone scorer
  vectorization.py         bezier fitting and control-point formatting
  segmentation.py          contour splitting
  region_vectorization.py  shared-border region engine
  island_generator.py      deterministic islands and label maps
  orchestration.py         config, batch runner, overlay, reports
  fileutils.py             SVG document writer
  colorspace.py            HSL palettes
  cvutils.py               alpha histogram, contour reshaping
  programutils.py          numeric helpers
  experimental/            research fitters, plots, original scratchpad
tests/
  conftest.py              scratch dir, sys.path, matplotlib cache
  helpers.py               shared line-art fixtures
  unit/                    one operational module at a time
  integration/             the real CLI and batch runner end to end
configs/                   sample orchestration configs
scripts/                   echo-only help pages
help.bat                   menu for those pages
resources/                 example inputs and reference SVGs
research/                  design notes
web-app/                   static shell and placeholder Flask mock
outputs/                   generated artefacts (gitignored)
```

Rules:

- `main.py` is a bootstrap only: it imports `build_parser`/`main` from
  `chunkmap.cli`. Put pipeline logic in the package, not in the entry file.
- Modules inside `chunkmap/` use **relative imports**
  (`from . import accuracy`, `from . import vectorization as vect`). Do not
  reintroduce flat `import accuracy` or `sys.path` manipulation; the package
  must work as `python main.py` and as `python -m chunkmap`.
- Tests import the application exactly like production does:
  `from chunkmap import accuracy`. Never import test helpers from a module
  inside `chunkmap/`.
- A unit test belongs in `tests/unit/test_<module>.py` and should exercise one
  module. Anything that calls `cli.main([...])`, reads/writes files, or crosses
  the vectoriser / generator / orchestration boundary belongs in
  `tests/integration/`.
- All scratch files go under `outputs/_test/`; the user profile may be read-only.
- `chunkmap/experimental/` is not production code. Nothing in the operational
  package may import it.

---

## 1. What the repository actually is

The product-level job:

> Take a map drawing (normally a PNG whose ink is in the alpha channel), turn it
> into clean SVG geometry, and measure how faithful that conversion was.

There are two vectorisation contracts:

| Mode | Input geometry | Output | Main use |
|---|---|---|---|
| **Ink-band** (legacy, default for plain `main.py INPUT`) | thresholded alpha / luminance mask of the drawn strokes | one closed `<path>` per traced contour, normally `fill:none; stroke:...` | hand-drawn line art, scans, arbitrary PNGs |
| **Region-aware** (`main.py --regions LABELS`, `orchestrate` with `region_mode: true`) | a single-channel integer label map (`0` water, positive = region) | one filled `<path>` per region; shared borders are the exact same cubic curves | generated islands, any map where region topology is known |

The second mode exists because the first one is geometrically honest but
visually wrong for subdivisions. A subdivision border is drawn as a stroke of
finite width. The ink-band mode traces **both sides of that stroke**, so two
regions get two parallel borders with the original line thickness between them.
Region mode ignores the stroke and uses the region partition itself, so adjacent
regions touch exactly and no line thickness is reproduced.

Do not “simplify” these into one mode. They answer different questions and have
different failure modes.

---

## 2. Non-negotiable user-visible capabilities

These are the behaviours a refactor must preserve. Each one is covered by tests,
but the tests are not a licence to delete the behaviour if a replacement passes
one golden case.

1. **Alpha ink with luminance fallback.** `accuracy.prepare_mask()` uses alpha
   when it varies; otherwise it flattens transparency onto white and uses
   luminance. Black-on-white scans and chalk-on-blackboard scans rely on this.
2. **Curvature-aware segmentation.** The ink-band path uses
   `segmentation.polygon_curvature_segmentation()` by default. `blind`
   segmentation exists as a fast baseline, not as an equivalent replacement.
3. **`max_chunk` is a hard ceiling.** A long smooth run is still cut so one cubic
   never spans a whole bay. This was a real regression once.
4. **Closed contours are joined, not chord-closed.** The last fitted cubic must
   meet the first one; otherwise SVG `z` draws a straight chord. This is what
   `controlPointFormatter(closed=True)` protects.
5. **Accuracy is a triple, not one number.** Tolerant IoU, strict IoU, and the
   exact-contour ceiling answer different questions. Boundary metrics measure
   both directions and report Hausdorff and within-1/2-px coverage.
6. **Polygon mode remains a baseline.** `--curve polygon` writes the exact raster
   contour. It is the reference against which the bezier fit is judged.
7. **The island generator is deterministic per seed and exposes its label map.**
   `IslandResult.labels` is the region geography, not a debug artefact.
8. **Region mode fits each shared boundary once.** Both adjacent regions reuse
   the same cubic segments in opposite directions. Independent per-region fitting
   is not an acceptable replacement.
9. **Region fitting clamps cubic endpoints to lattice junctions.** The free cubic
   fitter does not preserve endpoints by itself.
10. **Region chains are split at corners.** A single cubic through a sharp map
    corner overshoots wildly. `corner_angle` is not a cosmetic parameter.
11. **Region fills meet without a visible stroke.** The fill path has no outline;
    a same-colour hairline (`region_seam`) hides renderer antialiasing seams.
12. **Overlay pixels outside the raster ink are visible.** The overlay raises
    alpha where the vector line falls outside the island PNG; otherwise a bad
    curve would be invisible against the transparent background.
13. **Orchestration keeps generating after a single image fails** (unless
    `--fail-fast`), records the failure, writes a summary, and exits non-zero.
14. **Reports are machine-readable.** Per-image JSON, `summary.json`,
    `summary.csv`, and the schema keys are part of the integration contract.
15. **Exit codes are stable.** `0` success, `1` processing/threshold failure,
    `2` bad config/input, `130` interrupted.
16. **The `generate` and `orchestrate` subcommands are dispatched before normal
    argument parsing.** `python main.py orchestrate ...` must not be treated
    as an input file.
17. **The test suite writes only under `outputs/_test/`.** The sandbox/user
    profile may be read-only.

---

## 3. End-to-end data flows

### 3.1 Ink-band vectorisation

```
PNG
  -> accuracy.prepare_mask()          alpha > threshold, or luminance < threshold
  -> accuracy.trace_contours()        cv.findContours(RETR_CCOMP, CHAIN_APPROX_NONE)
  -> cli.collect_contours()           drop short / tiny contours
  -> segmentation                     curvature or blind runs
  -> vectorization.fitBezierCurve()   one cubic (4 control points) per run
  -> controlPointFormatter(closed=True)
  -> fileutils.SVGHandler             one <path> per contour
  -> accuracy.evaluate()              coverage + boundary + ceiling
```

Important detail: `accuracy.trace_contours()` returns pixel-centre chains. A
polygon through those points is about half a pixel inside the ink on every side.
That is why strict IoU looks lower than the visual fit and why the report also
computes the exact-contour ceiling.

### 3.2 Region-aware vectorisation

```
generator labels (int32)  or  --regions PNG/NPY
  -> region_vectorization.build_layout()
       build_boundary_edges()        pixel-lattice edges where labels differ
       trace_face_loops()            planar face walk, region kept on the left
       extract_chains()              split undirected graph at degree != 2
       fit_chain()                   fit each chain once; clamp endpoints
       _loop_to_cubics()             reuse chains, reverse for the other region
  -> RegionLayout
       paths: one RegionPath per region, one or more closed loops
       chains: fitted boundary chains (shared / coastline)
       exact_loops: unfitted lattice loops, used as the accuracy reference
  -> build_svg()
       style="fill"       one <path> per region, fill-rule="evenodd", no outline
       style="boundary"   one <path> containing every chain once, stroked
  -> accuracy.evaluate(mask=labels>0,
                       loops=vector region loops,
                       raster_contours=exact_loops)
```

The coordinate convention is **pixel centres**: a label-map pixel `(x, y)` is
the unit square `[x, x+1] x [y, y+1]`, and the vector points are shifted by
`-0.5` so a region polygon is centred on the pixel centres. The exact loops are
the same lattice loops before fitting.

### 3.3 Orchestration

```
config.yaml/json
  -> normalise_config()                 validate + fill defaults
  -> build_cases()                      balanced island/subdivision ranges
  -> for each case:
       island_generator.generate_island()
         -> islands/<name>.png
         -> regions/<name>.png          16-bit label map
       chunkmap.cli.main(...)           region mode or legacy mode
         -> svg/<name>.svg
       accuracy.evaluate()
         -> reports/<name>.json
       render_overlay()
         -> overlay/<name>.png
  -> reports/summary.json
  -> reports/summary.csv
  -> console per-image report + aggregate
```

The orchestration deliberately calls the real `cli.main()` instead of
reimplementing the converter. If you change the CLI, the integration test should
exercise it.

---

## 4. Module reference

### 4.1 `chunkmap/accuracy.py`

The measurement module and the standalone scorer.

- `describe_alpha(img)` — only a 4-channel image has alpha; `usable` means the
  channel actually varies.
- `prepare_mask(img, mode, alpha_threshold, luminance_threshold, invert)` —
  returns `(mask, resolved_mode, note)`. `auto` prefers alpha when usable.
  Luminance mode composites alpha onto white first; this is what makes
  black-on-white scans work.
- `trace_contours(mask)` — `cv.findContours(..., RETR_CCOMP, CHAIN_APPROX_NONE)`,
  returns pixel-centre `(n,2)` int arrays and the hierarchy.
- `sample_cubic`, `ctrl_to_loop` — flatten the control-point layout produced by
  `vectorization.controlPointFormatter`.
- `parse_path_data(d)` — supports absolute/relative `M`, `L`, `H`, `V`, `C`, `Z`.
  `S`, `Q`, `T`, `A` raise. Do not silently skip unsupported commands.
- `parse_svg(text)` — regex extraction of width/height/path `d`; returns paths,
  curve count, and flattened loops.
- `loops_to_mask(loops, shape)` — XORs each filled loop. This is the even-odd
  reconstruction used for both coverage and holes.
- `mask_scores(reference, candidate, tolerance)` — strict IoU/Dice/precision/
  recall plus a tolerant block where pixels within `tolerance` px of the other
  mask count as covered.
- `boundary_scores(raster_contours, loops, shape)` — draws both boundaries as
  1-px lines and measures distances in both directions, plus Hausdorff and
  within-1/2-px percentages. `raster_to_vector` is the one that answers “is the
  curve on the ink?”; `vector_to_raster` catches curves that bulge away.
- `evaluate(mask, loops, raster_contours, segments, control_points, tolerance)` —
  returns coverage, baseline, relative-to-exact-contours, boundary and
  compression. The baseline refills the **exact raster contours** to establish
  the ceiling under the same metric.
- `format_report(metrics)` — six-line human report. The orchestration reuses it,
  so CLI and batch output stay consistent.

Semantics that are easy to break:

- `relative_to_exact_contours.strict_iou` is a **ratio**, not a percentage.
  `format_report` multiplies by 100.
- It can exceed `1.0` because `cv.fillPoly` treats boundary pixels inclusively.
  Region mode and `orchestration.compact_metrics()` clamp it to 1.0; do not
  change the metric formula to hide this.
- `compression = raster_points / segments`; `segments` is the curve count, not
  the control-point count.
- `boundary.within_1px_pct` is the percentage of **raster points** within one
  pixel of the vector, not the percentage of pixels in the image.

### 4.2 `chunkmap/cli.py` and root `main.py`

The root `main.py` is a three-line convenience entry point. The parser and all
command implementations live in `chunkmap/cli.py`, which is what the tests and
orchestration import. Single entry point for:

- `main.py INPUT` — ink-band conversion;
- `main.py generate ...` — random island PNGs;
- `main.py orchestrate ...` — batch integration test;
- `main.py INPUT --regions LABELS` — region-aware conversion.

Key structure:

- `main(argv)` checks `generate` and `orchestrate` **before** `argparse`. Keep
  this order.
- `_load_pipeline()` imports the heavy vectorisation stack lazily. `generate`
  works without matplotlib; `orchestrate` sets `MPLCONFIGDIR`/`MPLBACKEND`
  before importing the CLI module, because `fileutils` imports matplotlib.
- `collect_contours()` preserves contour depth (outer/hole) for verbose output.
- `segment_contour()` / `vectorize_contour()` wrap the existing fitter.
- `build_svg()` creates `SVGHandler`, palette classes, and paths.
- `run_region_mode()` is the separate region path. It loads the label map,
  builds a layout, writes the SVG, and scores against `labels > 0` with the
  exact lattice loops as the boundary reference.
- `--report-json` payloads are consumed by humans and by scripts; preserve the
  names.

Exit codes: `0`, `1`/`2` per the guardrail section; `130` on interrupt.

### 4.3 `chunkmap/region_vectorization.py`

This is the shared-border engine. Read its module docstring as well.

Data model:

- `BoundaryEdge` — one undirected unit edge of the label boundary graph.
- `Chain` — a boundary path between junctions (or a closed coastline), with
  `lattice`, simplified `points`, `edge_ids`, the two `labels`, `closed`, and
  fitted `cubics`.
- `RegionPath` — one region id, one or more loops of cubics, area/perimeter.
- `RegionLayout` — labels, edges, chains, paths, `exact_loops`, and counts.

Algorithm details:

1. **Boundary edges** are the lattice edges where 4-neighbour labels differ.
   Canvas edges become water boundaries. No stroke width is involved.
2. **Directed half-edges** sample the label on the left of each direction.
3. **Face traversal** starts from each directed edge whose left region is the
   target, then at each junction chooses the first outgoing edge clockwise from
   the reverse of the incoming edge. This keeps the region on the left and
   handles 3- and 4-region junctions without special cases.
4. **Chains** are the undirected graph split at vertices with degree != 2. A
   closed coastline with no junctions remains one closed chain.
5. **Fitting** uses the existing `vectorization.fitBezierCurve` on slices of the
   simplified chain. `_open_slices()` splits at windowed turns >=
   `corner_angle_deg` and enforces `max_chunk`. Endpoint clamping is mandatory:
   `fitBezierCurve` is a free total-least-squares fit and its endpoints can drift
   by fractions of a pixel.
6. **Region loop assembly** rotates a face loop to a chain boundary, groups
   consecutive directed edges by chain, verifies the whole chain is used, and
   reverses the cubic list (including each cubic's control points) when the
   chain is traversed the other way.
7. **Exact loops** are kept for accuracy. They are the same face loops before
   fitting, shifted by `-0.5`.
8. **SVG** has one path per region for `fill`; shared chain cubics are reused, so
   borders are identical. `seam_width` adds a same-colour hairline, not a visible
   outline. `boundary` style emits every chain once.

Failure modes deliberately raise `RegionVectorizationError`:

- a face loop does not close;
- a chain changes direction inside one region loop;
- a region loop uses only part of a chain;
- two consecutive chains do not share an endpoint;
- a region has no boundary loop.

Do not “fix” these by snapping or by falling back to independent region masks.
That reintroduces the exact seam problem this module exists to remove.

### 4.4 `chunkmap/island_generator.py`

The deterministic random island generator.

Pipeline: silhouette -> signed distance field -> warped multi-octave Perlin fBm
-> coverage-solved sea level -> cleanup -> mask -> region labelling -> render.

Key facts:

- `IslandParams` is frozen; all random state comes from one
  `np.random.default_rng(seed)` plus per-octave permutations.
- `coverage`, `margin`, `islands`, `subdivisions`, `filled`, `stroke_width` are
  meaningful API. `style_preset()` exposes the named presets.
- `IslandResult.mask` is the **land** mask. `IslandResult.labels` is the region
  map; `labels > 0` is exactly `mask > 0`.
- `_label_and_subdivide()` creates one label per island when subdivisions=0, and
  area-aware Voronoi regions otherwise. Small regions are merged; the reported
  `region_count` is the actual count, not `islands * subdivisions`.
- `_render_image()` draws:
  - outline mode: a constant-width band around the coastline and every label
    boundary, with antialiased alpha;
  - filled mode: an opaque land fill plus darker boundary lines.
- The outline alpha is normalised to its peak. This is good visually but makes
  the converter default `alpha > 252` fracture the thin band. Region mode avoids
  the issue entirely; legacy conversion needs `alpha_threshold` around 128.
- The generator may reduce coverage to keep multiple islands separate. The
  returned `coverage` is the real coverage, not the target.
- Islands can still touch at high counts/coverage; label topology then reflects
  the touching regions. The orchestration reports generator counts, but cannot
  invent separation that the raster does not have.

### 4.5 `chunkmap/segmentation.py`

Closed-contour segmentation. `polygon_curvature_segmentation()`:

- computes curvature after Gaussian smoothing (`compute_curvature` wraps the
  first three points, so it assumes a closed contour);
- splits at high-curvature points;
- merges tiny segments;
- enforces `max_chunk` by inserting evenly spaced splits.

`equidistant_splits()` clamps to the last valid index. The regression tests cover
the degenerate splits that used to crash the fitter. Do not remove those clamps.

### 4.6 `chunkmap/vectorization.py`

- `polygonSegmentation(data, chunk_size)` — blind segmentation; public baseline.
- `fitBezierCurve(data, use_cache=True)` — Pastva-style TLS + Gauss-Newton fit.
  Returns a `(4,2)` control-point array for a cubic, but **does not guarantee
  that the first/last points equal the input endpoints**. Region fitting clamps
  them; ink-band fitting joins midpoints with `controlPointFormatter`.
- `controlPointFormatter(ctrlPoints, closed=True)` — converts fitted cubics to
  SVG rows. It averages the endpoint between adjacent curves (midpoint join) and
  when `closed=True` joins the last curve back to the first. Without this, SVG
  `z` closes every contour with a straight chord. Do not replace it with a raw
  concatenation in ink-band mode.
- `chunkmap/experimental/bezier.py` and `bezier_improved.py` are research
  fitters/benchmarks. They are not on the CLI path.
  `chunkmap/experimental/graphutils.py` is diagnostic plotting.
  `chunkmap/experimental/legacy_main.py` is the old hand-driven scratchpad. Do
  not wire them in without tests and a reason.

### 4.7 `chunkmap/fileutils.py`

- `SVGHandler` — header, style tag, path emitters, compose.
- `drawcubicbezier()` — one closed path from control-point rows.
- `saveCons()` — legacy filled-polygon writer at 75% opacity. It is not the CLI
  writer and does not set even-odd holes.
- `createFolder()` / `save()` — date-stamped folders under `cwd`. They are
  historical; the CLI avoids them for controlled output paths. The module still
  imports matplotlib, which is why orchestration sets a writable `MPLCONFIGDIR`
  before importing it.

### 4.8 `chunkmap/colorspace.py`, `chunkmap/cvutils.py`

- `colorspace.cmapHex()` generates the palette used by `--fill palette` and
  region fills.
- `cvutils.histimg()` requires genuine 4-channel RGBA; `accuracy.prepare_mask`
  is the more general mask helper. `conReshaper()` is used before fitting.

### 4.9 `chunkmap/orchestration.py`

Config loader, case planner, batch runner, overlay renderer, reporting.

- Loader accepts JSON and YAML. PyYAML is optional; otherwise a built-in subset
  parser handles nested mappings/sequences, inline lists/maps, quoted scalars,
  comments, and plain scalars (including commas). No anchors/tags/block scalars.
  UTF-8 BOMs are accepted.
- `normalise_config()` is strict about unknown keys, ranges and cross-field
  constraints. Error messages name the path.
- `build_cases()`:
  - explicit `generation.images` wins over `count`;
  - otherwise the island and subdivision ranges are **shuffled and cycled** so a
    small sample covers every value, instead of accidentally drawing one value;
  - each image seed is `seed + (index-1) * seed_step`;
  - names are sanitised and unique.
- `run()` writes `islands/<name>.png`, `regions/<name>.png` (16-bit label map),
  `svg/<name>.svg`, `overlay/<name>.png`, `reports/<name>.json`, then
  `summary.json` and `summary.csv`.
- Region mode passes the label map to the real CLI with `--regions`; legacy mode
  passes the ink-band options. Both are scored against the appropriate
  reference (`labels>0` vs the alpha/luminance mask). `region_palette: spread`
  uses `region_vectorization.default_palette()`; `region_palette: hue` uses the
  legacy `--hue` palette.
- `render_overlay()` draws parsed SVG loops in a contrasting colour, optionally
  tinting the even-odd fill. It composes in premultiplied alpha and raises output
  alpha where the vector falls outside the base ink.
- `_aggregate()` computes mean/median/min/max per metric, best/worst IoU, stage
  timings and totals. `compact_metrics()` clamps `relative_iou` to 1.0.
- Failure policy: one image failure is recorded and the batch continues; the
  summary is still written; exit 1. `--fail-fast` stops after the first.
- `--clean` removes only the managed subfolders (`islands`, `regions`, `svg`,
  `overlay`, `reports`) inside the test space. Never widen this to the space
  itself or to arbitrary paths.

### 4.10 `web-app/app.py`, `web-app/`

A Flask mock and a static browser shell. The endpoints return random
rectangles/circles; they are **not** connected to the vectoriser. Do not describe
them as a working upload/download app. They are scaffolding for a future UI.

---

## 5. Accuracy semantics in one page

Given a raster ink mask `M` and vector loops `V`:

- **Coverage (tolerant IoU)** — fill `V` with even-odd, dilate each mask by
  `tolerance` px, and score. This is the headline number: did the vector land on
  the ink? One pixel is the default tolerance.
- **Coverage (strict IoU)** — compare filled areas exactly. It is biased low for
  thin strokes because the raster contour runs through pixel centres. Never use
  strict IoU alone to judge a fit.
- **Ceiling** — refill the exact raster contours and score that against `M`. It
  is the best any vector approximation of those contours can do under the same
  strict metric. The reported ratio is `strict_iou / ceiling`.
- **Boundary raster→vector** — for each raster contour point, distance to the
  nearest flattened vector curve. This is the “is the curve on the ink?” number.
- **Boundary vector→raster** — the reverse; catches overshoot away from the ink.
- **Hausdorff** — worst of the two directions.
- **within 1/2 px** — share of raster boundary points within that distance.
- **Compression** — raster points / cubic segments.

Region mode uses the exact lattice loops as `raster_contours`, not the land
silhouette. Using the silhouette would make every internal shared border look
like a huge boundary error, because it has no internal borders. The coverage
reconstruction still uses `loops_to_mask`'s XOR rule; two adjacent region fills
can cancel a boundary pixel, so region strict IoU is slightly conservative. The
tolerant IoU is the headline number and absorbs that pixel; do not change XOR to
OR just for region mode without keeping the ink-band hole behaviour intact.

---

## 6. Tests and what they protect

Run:

```bat
venv\Scripts\activate
python -m pytest -q
```

The suite writes only under `outputs/_test/`.

Unit tests (`tests/unit/`) exercise one module; integration tests
(`tests/integration/`) drive the real CLI across module boundaries.

| File | Protects |
|---|---|
| `tests/unit/test_accuracy.py` | mask modes, alpha description, SVG path parser, even-odd reconstruction, mask scoring |
| `tests/unit/test_vectorization.py` | degenerate fits, closed formatter, blind segmentation |
| `tests/unit/test_segmentation.py` | curvature splits, `max_chunk` ceiling, equidistant clamp |
| `tests/unit/test_fileutils.py` | SVG header/style/path composition, folder creation |
| `tests/unit/test_colorspace.py` | hex conversion, palette length, seeded reproducibility |
| `tests/unit/test_cvutils.py` | alpha histogram validation, contour reshaping |
| `tests/unit/test_island_generator.py` | determinism, size/coverage/margin, outline vs filled, multiple islands, subdivisions |
| `tests/unit/test_region_vectorization.py` | shared border, exact closure, 3-/4-region junctions, holes, SVG fill/boundary styles, label PNG round-trip |
| `tests/unit/test_orchestration_config.py` | YAML/JSON config, BOM, strict validation, case planning, overlay compositing |
| `tests/integration/test_cli_convert.py` | mask errors, polygon baseline, ink-band end to end, report JSON |
| `tests/integration/test_cli_generate.py` | `generate` CLI paths |
| `tests/integration/test_cli_regions.py` | region CLI, shared-chain report, wrong-size label map |
| `tests/integration/test_orchestration_run.py` | full batch, legacy mode, thresholds, failure continuation, `--only` |

When you add a module, add `tests/unit/test_<module>.py`. When you add a command,
add an integration test that calls `cli.main([...])`.

If you add a new vectorisation mode, add a test that proves its shared-border or
coverage contract. If you change a metric, update the README explanation and the
sample output, not just the test thresholds.

---

## 7. Performance profile

Approximate on the development machine (Python 3.13, OpenCV, SciPy, no GPU):

- Island generation: ~1.5–3 s for 1024×768 with 1–4 islands and 0–4
  subdivisions.
- Ink-band fitting: often the dominant cost; ~5–20 s per 1024×768 image
  depending on contour count and `max_chunk`.
- Region fitting: similar or a little faster because it fits boundaries once,
  not both sides of each stroke. A 16-region 1024×768 map is roughly 10–15 s of
  layout/fitting with `max_chunk=10, corner_angle=55`.
- Accuracy scoring: usually 0.05–0.3 s per image at 1024×768.
- Overlay: ~0.1–0.3 s per 1024×768 image.

Do not “optimise” by lowering `max_chunk` or by fitting one cubic per region
without measuring both the visual result and the accuracy report.

---

## 8. Known limitations

- Region mode requires a label map. It does not infer region topology from a
  stroke band. Inferring whether a line separates two regions or is a coastline
  is ambiguous without extra information; the generator supplies the labels.
- Ink-band mode still traces both sides of a stroke. That is correct for
  hand-drawn line art but wrong for a subdivision map. Use `--regions`.
- The SVG writer does not use layers/groups; region mode uses one flat path per
  region with a class.
- Region holes are supported as multiple loops in one `evenodd` path; the
  generator currently fills holes, so this is mostly for generic label maps.
- Region labels are assumed to be contiguous enough for a clean planar graph.
  Isolated single-pixel labels or pathological checkerboards may raise a layout
  error; that is preferable to silently emitting bad borders.
- `fileutils` still has `cwd`-relative helpers used by legacy code. New code
  should pass absolute paths.
- `web-app/app.py` and the rest of `web-app/` are not a working product path.
- The overlay is a diagnostic drawing, not a browser render of the SVG styles.
  It draws every parsed loop as a closed stroke (and optional even-odd fill) in
  the overlay colour.

---

## 9. Refactoring guardrails (read this twice)

### Do not delete these “optimisations”

1. **Lazy imports and `MPLCONFIGDIR`.** Without them, a read-only user profile
   makes matplotlib print warnings or fail while caching fonts.
2. **The balanced range shuffle.** It is not random decoration; it makes a small
   sample cover all island/subdivision counts.
3. **The BOM-tolerant config read.** Windows editors and PowerShell add BOMs.
4. **The YAML subset parser.** It keeps YAML usable without adding a dependency.
   PyYAML is optional, not required.
5. **The region label sidecar.** It is the input to shared-border vectorisation.
6. **The face-walk direction rule.** It is what makes 3-/4-region junctions
   produce correct regions without special cases.
7. **The chain split at degree != 2.** If you fit across junctions, shared
   geometry is lost.
8. **The endpoint clamp.** Without it, adjacent chains have subpixel gaps.
9. **The corner split.** Without it, a cubic through a 90° map corner overshoots
   by tens of pixels.
10. **The `max_chunk` enforcement** in both segmentation and region fitting.
11. **The seal between adjacent fills** (`region_seam`). Without it, some SVG
    renderers show a faint hairline between two exactly coincident fills.
12. **The overlay alpha boost.** A curve outside the ink must not disappear.
13. **The failure-continuation policy.** A single bad seed should not lose a
    whole sample.
14. **The report schema and CSV columns.** Downstream scripts and the README
    depend on them.

### Do not “simplify” these away

- Do not replace curvature segmentation with `blind` because it is faster.
- Do not replace the tolerant/strict/ceiling triple with a single IoU.
- Do not replace the exact-contour ceiling with the input mask.
- Do not remove `--curve polygon`; it is the reference baseline.
- Do not remove `--fill palette`; palettes are a real product feature.
- Do not remove `IslandResult.labels` as “unused” because the renderer does not
  need it; region mode does.
- Do not unify `region_fill` with `fill`; they belong to different modes.
- Do not merge `region_seam` into `stroke_width`; the seam is same-colour and is
  not a visible outline.
- Do not reimplement `cli.main()` inside orchestration for speed; the
  orchestration is an integration test, not a second converter.
- Do not treat `chunkmap/experimental/bezier.py`, `bezier_improved.py`,
  `graphutils.py` or `legacy_main.py` as the production path. They are
  research/scratch material.
- Do not claim the Flask/web shell is functional; it is not wired.

### When changing geometry

- Always check both `tests/unit/test_region_vectorization.py` and
  `tests/integration/test_cli_convert.py`.
- Run the sample orchestration with `--count 2` and inspect:
  - a region SVG in a browser or SVG viewer,
  - the overlay PNG,
  - `summary.csv`.
- A change that raises coverage but introduces a visible border gap is a
  regression. A change that lowers strict IoU but keeps tolerant IoU and exact
  shared borders may be correct.
- If you replace a fitter, verify that endpoints are preserved or re-clamp them.

### When changing configuration

- Add the key to `_VECTOR_KEYS`/`_GENERATION_KEYS`/etc. and to validation, or it
  will be rejected as unknown.
- Add the default to the dataclass and `to_dict()`; the summary snapshot must
  reproduce the run.
- Update the sample YAML **and** JSON; they are expected to be equivalent.
- Update `README.md`, this file, and the sample output if user-visible output
  changes.

---

## 10. Verification recipes

Fast unit/integration suite:

```bat
python -m pytest -q
```

Ink-band smoke test:

```bat
python main.py resources\example-cuba.png --report
python main.py resources\example-cuba.png --curve polygon --report
```

Region smoke test:

```bat
python main.py output\island.png --regions output\island-labels.png ^
    --region-style fill --report-json output\island-regions.json
```

Orchestration smoke test:

```bat
python main.py orchestrate --config configs\island-svg-sample.yaml ^
    --count 2 --space outputs\orchestration\smoke --clean
```

Inspect `reports/summary.csv`; the region run should show high tolerant IoU and
`shared_chain_count > 0` for maps that have subdivisions.

---

## 11. Glossary

- **Ink mask** — binary raster of pixels considered drawn.
- **Label map** — integer raster; `0` water, positive region ids.
- **Region** — one connected labelled land area.
- **Boundary edge** — one unit lattice edge between two different labels.
- **Half-edge** — directed boundary edge with a known region on its left.
- **Face walk** — traversal of a region boundary using the next-edge rule.
- **Junction** — boundary vertex with degree != 2.
- **Chain** — fitted boundary path between junctions, or a closed coastline.
- **Shared chain** — chain with land on both sides; reused by both regions.
- **Coastline chain** — chain with land on one side and water on the other.
- **Exact loop** — unfitted pixel-lattice region loop.
- **Tolerant IoU** — IoU with a pixel tolerance; headline coverage metric.
- **Ceiling** — strict IoU of the exact raster contour refilled; the maximum a
  vector approximation of those contours can reach under the strict metric.
- **Test space** — the directory orchestration writes generated inputs and
  reports into.
- **Overlay** — island PNG with parsed SVG paths drawn in a contrasting colour,
  for visual inspection.

---

## 12. Documentation map

- `README.md` — user-facing commands, config schema, sample output, accuracy
  explanation.
- `research/01..05` — design research for the island generator (silhouette, SDF,
  noise, integration/evaluation). Useful when changing generation, not the
  vectorizer.
- `AGENTS.md` (this file) — architectural contract and refactor guardrails.
- Module docstrings in `chunkmap/accuracy.py`,
  `chunkmap/region_vectorization.py`, `chunkmap/island_generator.py`, and
  `chunkmap/orchestration.py` — the closest executable documentation of each
  stage.
- `tests/unit/` and `tests/integration/` — the executable contract; the unit
  file names mirror the module names.
- `configs/island-svg-sample.yaml` / `.json` — canonical config shape.
- `scripts/help-*.bat` — the README commands in echo-only form.

If this file and the code disagree, inspect the tests and the module docstrings;
then update this file. Do not silently follow an outdated paragraph.
