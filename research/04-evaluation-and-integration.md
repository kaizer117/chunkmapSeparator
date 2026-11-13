# 04 — Evaluation and Integration

The generator is only useful if "looks natural" can be checked without arguing about
taste every time. This document defines acceptance criteria, metrics, a test matrix
and the integration checks against the existing raster-to-SVG pipeline.

The first implementation now exists in `code/island_generator.py`, but the test
matrix and visual review below have not been run in full. This is the scoring plan
for that validation.

---

## 1. Acceptance criteria

### Hard requirements (must pass)

| Criterion | Target | How measured |
|---|---|---|
| Deterministic | same seed + params → identical mask | SHA-256 of the mask bytes |
| Non-empty | at least one component ≥ `min_island_area` | connected components |
| Component count | `1` for single-island mode; `N` for archipelago mode | connected components |
| Coverage | within ±2% of `coverage` | mask area / short-side square area |
| Margin | no land within `margin` of any border | bounding box + per-edge check |
| Simple outline | no self-intersections | segment-intersection test on the final polygon |
| Minimum width | no land thinner than `2 px` at output size | distance transform / skeleton |
| Hole policy | no holes when `lagoons=false`; holes ≥ `min_hole_area` when true | contour hierarchy |
| Valid SVG | every path closes; no NaN/inf; parseable | existing SVG parser / renderer |
| No crash | all seeds in the test matrix produce output or a clear rejection | test harness |

### Soft requirements (quality targets)

| Criterion | Target | Why |
|---|---|---|
| Coast fractal dimension | `D ≈ 1.15–1.40` | in the range of natural coasts |
| Compactness `4πA/P²` | `0.15–0.65` | avoids both circles and confetti |
| Convexity `A / A_hull` | `0.55–0.95` | enough bays, not a star |
| Diversity across seeds | mean pairwise IoU `< 0.65` | avoids "same island, different noise" |
| Curvature | no long straight runs; no 90° corners | vector quality |
| Visual score | mean ≥ 3.5 / 5 over 30 samples | human check |
| Runtime (generation only) | `< 100 ms` at 512², `< 500 ms` at 1024² | interactive tuning |
| Runtime (end-to-end) | comparable to existing CLI on the same size | integration |
| Detail consistency | no visible change in character at 2x output scale | scale test |

The fractal dimension range is a guide, not a gate. A stylised cartographic preset
may deliberately sit at `D ≈ 1.1`, and a fjord preset at `D ≈ 1.5`; those are valid
if they match the style.

---

## 2. Metrics

### 2.1 Area and coverage

Coverage is measured against the short-side square, not the full canvas (see
[03-recommended-pipeline.md](03-recommended-pipeline.md#51-solving-for-coverage)):

```
A_ref = min(W, H)^2
area_error = |A_land - coverage * A_ref| / (coverage * A_ref)
```

This makes the target comparable across 16:9, 1:1 and 9:16 canvases. If a
canvas-area basis is used instead, change `A_ref` to `W * H` and report both.

Compute on the final mask, after cleanup and before/after smoothing. If smoothing
shrinks the area, report both; the coverage solve should target the post-smoothing
area, or compensate by a small offset.

### 2.2 Margin and bounding box

```
margin_px = min(left, top, W - 1 - right, H - 1 - bottom)
```

where `left/top/right/bottom` are the mask bounding box. The target is
`margin_px >= margin * min(W, H)`. Also check each edge separately, because a
bounding box can be inside the margin while one coast still touches an edge.

### 2.3 Components and holes

Use 8-connectivity:

```
num, labels, stats, _ = connectedComponentsWithStats(mask, 8)
min_area_px = min_island_area * min(W, H)^2
components = count(stats[i].area >= min_area_px)
```

Holes are background components not connected to the border. Count them and their
areas. Report `holes_total`, `holes_kept`, `largest_hole_area`.

### 2.4 Minimum width

A thin isthmus can survive connected-component counting and still look wrong or
break during smoothing. Estimate the local width from the distance transform:

- Compute `D = distanceTransform(mask)`, where `D[p]` is the distance to the nearest
  water pixel.
- The local width at a skeleton point is approximately `2 * D`.
- If a skeleton is available, take the 5th percentile of `2 * D` along it; if not,
  use a morphological erosion test: erode the mask by a disk of radius `r`; if the
  component disappears before `r = min_width/2`, it is too thin.

A useful but easily confused quantity is the largest inscribed circle:
`diameter = 2 * max(D)`. That is not the minimum width. For a first pass, use
erosion at `r = min_width/2` and check that a component remains.

### 2.5 Compactness and convexity

```
compactness = 4 * pi * A / P^2          # 1 for a circle, lower for irregular
convexity   = A / area(convex_hull)
```

Both are cheap and useful as sanity checks, but neither is a naturalness metric on
its own. A circle has perfect compactness and convexity; a natural island is
somewhere in the middle.

### 2.6 Fractal dimension

Box counting on the coastline:

1. Rasterise the outline (1 px wide) or use the contour points.
2. For scales `eps = 2, 4, 8, 16, 32, 64` px, count the number of occupied boxes
   `N(eps)` that contain at least one outline pixel.
3. Fit a line to `log N(eps)` vs `log(1/eps)`; the slope is `D`.

Caveats:

- The range of scales must be several octaves; with only 3 points the fit is noisy.
- Pixel rasterisation adds a floor at small scales; ignore `eps < 2 px`.
- The measured `D` depends on the contour resolution and smoothing; compare runs
  with the same pipeline, not against a textbook number.
- For a quick alternative, measure perimeter at several RDP tolerances and fit
  `log P` vs `log(1/eps)`; the slope is related to `D` but not identical.

### 2.7 Curvature and corner statistics

On the smoothed polygon, compute turning angles between consecutive segments:

- count angles above `60°` (sharp corners) per 100 points;
- count runs of nearly collinear points (long straight sections);
- compute the mean absolute curvature.

The goal is not a specific number; it is to catch the two failure modes: pixel stair
steps (many 90° angles) and over-smoothing (almost no curvature variation).

### 2.8 Diversity

For `M` seeds with the same parameters, compute pairwise IoU of the binary masks:

```
intersection = count(M_i & M_j)
union        = count(M_i) + count(M_j) - intersection
IoU_ij       = intersection / union
```

Report the mean and the maximum. A generator that always produces the same blob with
different noise will have a high mean (> 0.8); a good generator should be well below
that. The target `< 0.65` is a starting point, not a hard rule; small islands have
higher IoU by construction because there is less shape to vary.

### 2.9 Determinism

```
hash = SHA-256(mask.tobytes())
```

Run the same seed three times in the same process and in a fresh process. They must
match. If the noise library uses a global RNG, reseed it explicitly and avoid
depending on hash randomisation (`PYTHONHASHSEED`).

### 2.10 Runtime and memory

Measure separately:

- noise generation;
- distance transform;
- threshold + cleanup;
- contour + simplify + smooth;
- SVG/Bézier fit.

Use `time.perf_counter`, warm up once, report median of 5 runs. Peak memory with
`tracemalloc` or the OS. At 1024², the field arrays are a few MB each; the
distance transform and labels are the largest temporary allocations.

### 2.11 Visual review

Metrics cannot decide whether an island looks good. Use a small blind review:

- 30 images per style, randomised order, no labels.
- 3–5 reviewers, 5-point Likert: 1 = blob/artificial, 5 = natural.
- Ask specifically: "Does this read as an island? Does the coast have bays and
  capes? Is there any obvious grid/noise artefact?"
- Keep the worst-scoring seeds as regression cases.

A useful trick: show the images at thumbnail size (128 px) as well as full size. A
silhouette that only works at full resolution is not a good island.

---

## 3. Test matrix

The matrix is intentionally small enough to run during development and broad enough
to catch the known failure modes.

### 3.1 Sizes and aspects

| Case | Size | Why |
|---|---|---|
| small square | 256×256 | fast iteration; detail limits |
| default square | 1024×1024 | main quality target |
| landscape | 1920×1080 | aspect correction, margin |
| portrait | 1080×1920 | aspect correction |
| large map | 4096×3072 | scaling, detail, performance |
| extreme wide | 2048×512 | margin/aspect stress |

At least one case should generate at 1024 and scale to 4096 to check that the
character survives upscaling.

### 3.2 Shape parameters

| Parameter | Values |
|---|---|
| `coverage` | 0.10, 0.30, 0.50 |
| `aspect` | 0.5, 1.0, 2.0 |
| `lobes` | 2, 3, 5 |
| `hurst` | 0.62, 0.74, 0.82 |
| `warp` | 0.0, 0.15, 0.30 |
| `coast_amp` | 0.05, 0.12, 0.20 |

Do not run the full cross-product; use one-factor-at-a-time slices around the
default, plus a few combined "preset" cases.

### 3.3 Topology and style presets

- Single natural island.
- Smooth/cartographic island.
- Rugged/fjord island.
- Archipelago with 5 islands.
- Atoll/lagoon.
- Barrier island.
- One "pathological" case: `coverage=0.55`, `aspect=3`, `warp=0.3` (should reject or
  degrade gracefully, not crash).

### 3.4 Seeds

- 20 seeds for visual review per style.
- 50 seeds for metric distributions (coverage, component count, diversity).
- 5 fixed "golden" seeds stored as regression fixtures once the generator is stable.

### 3.5 Pipeline variants

| Variant | Purpose |
|---|---|
| Perlin vs OpenSimplex vs value noise | noise quality and speed |
| `findContours` vs marching squares | contour quality and half-pixel bias |
| no smoothing vs 1 Chaikin vs 2 Chaikin | over/under-smoothing |
| `simplify_px` 0.3 / 0.6 / 1.2 | bay preservation |
| with/without coastal band | interior cleanliness and coast detail |

---

## 4. Experiment plan (when implementation starts)

**Experiment A — shape space.** 5 presets × 20 seeds at 512². Review thumbnails and
full size. Pick the two best presets; discard parameters that consistently produce
blobs or confetti.

**Experiment B — noise comparison.** Same base shape and parameters, three noise
primitives. Measure runtime, visual score, and the number of axis-aligned artefacts
in the contour. Pick one for production, keep one fallback.

**Experiment C — contour comparison.** `findContours` vs marching squares on the same
field. Measure:
- number of contour points;
- SVG size after Bézier fitting;
- boundary distance to the source mask;
- visible stair steps at 2x zoom.

**Experiment D — smoothing.** 0/1/2 Chaikin, with and without RDP, scored by:
- area shrinkage;
- bay preservation (count of concavities above a depth threshold);
- visual smoothness;
- self-intersections.

**Experiment E — scale.** Generate at 512, 1024 and 2048; compare the silhouettes at
the same display size. Check that the character (not just the outline) is stable.

**Experiment F — integration.** Convert 10 generated masks through `code/cli.py`;
record time, curve count, SVG size, coverage IoU and boundary error. Compare with the
existing hand-drawn examples.

---

## 5. Integration checks with the existing pipeline

### 5.1 Mask format

- RGBA PNG, alpha 255 on land, 0 on water.
- The CLI's default `alpha > 252` threshold sees the interior as ink.
- If the mask is antialiased, either output a hard mask or run the CLI with a lower
  alpha threshold / `--mask luminance` on a black-on-white version.
- Do not use a transparent background with coloured land unless the alpha is fully
  opaque; the CLI uses alpha, not colour.

### 5.2 Contour and fit

- `code/cli.py` uses `cv2.findContours` + curvature segmentation + Bézier fitting.
- A generated contour with thousands of pixel-level points can produce many curve
  segments and a slow fit. The pre-smoothing/RDP step is important for fit time as
  well as appearance.
- Check the CLI report: `coverage IoU` should be high (the vector should match the
  mask), and `boundary mean` should be well under 1 px.
- The fitter can overshoot in tight bays; compare the rendered SVG at 2x zoom, not
  just the numbers.

### 5.3 Holes

- Outline-only: holes are just extra contours; fine.
- Filled: the current writer does not use `fill-rule="evenodd"`. A lagoon will be
  filled as land unless the writer is changed or the lagoon is painted as water on
  top.
- Decide the hole policy before wiring the generator into the UI.

### 5.4 Accuracy tool

`code/accuracy.py` can score a generated SVG against the generated mask:

```
python code/accuracy.py island_mask.png island.svg --json report.json
```

Use it as a regression check:
- coverage IoU (1 px tolerance) > 0.98;
- boundary mean < 1 px;
- no parse errors;
- SVG size within a reasonable factor of the hand-drawn examples.

This measures the *conversion*, not the naturalness of the island. Both are needed.

### 5.5 Web app

If this becomes a UI feature:

- generate in a background job, not on the request thread;
- cache the parameter set + seed + result;
- return the SVG (or a PNG preview) with the same dimensions as the canvas;
- provide a seed field so a user can reproduce an island;
- debounce parameter changes; 1024² generation should be fast enough for a slider,
  but the Bézier fit may not be.

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Parameters interact unpredictably | high | medium | presets, one-factor-at-a-time tuning, live preview |
| "Natural" is subjective | high | medium | blind visual review, multiple styles, keep worst cases |
| Topology instability (fragments, holes) | medium | high | component/hole cleanup, validation, retry |
| Edge contact | medium | high | inset base, soft edge penalty, retry |
| Over-smoothing kills character | medium | medium | compare 0/1/2 passes; keep `simplify_px` small |
| Noise library is slow | medium | medium | benchmark; generate at 512 for preview; vectorised array API |
| Grid artefacts from Perlin/value noise | medium | medium | rotate octaves, offset, domain warp, prefer OpenSimplex |
| 4K output loses detail | medium | low | generate at 1024/2048; scale path; document the limit |
| Fill-rule bug with lagoons | high (if filled) | high | outline-only first; fix writer separately |
| Determinism breaks across versions | low | medium | explicit seeds; golden mask hashes; pin dependency versions |
| Scope creep into full terrain | medium | medium | keep this phase outline-only; heightfield is a foundation, not a deliverable |

---

## 7. Definition of done (for the implementation phase)

The research is done when the algorithm choice and parameters are justified. The
implementation is done when:

1. The hard requirements in §1 pass for every case in the test matrix.
2. 20 seeds per style score at least 3.5/5 in the blind review, with no style
   producing repeated identical silhouettes.
3. Coverage is within ±2%, margin is respected, and there are no unintended
   fragments or holes.
4. The generated mask converts through the existing CLI with coverage IoU > 0.98
   (1 px tolerance) and boundary mean < 1 px.
5. Generation at 1024² is under 500 ms (generation only) on the development machine.
6. The same seed produces the same mask across runs and processes.
7. The parameter set and presets are documented, and a seed + params can reproduce
   an island exactly.
8. The SVG is valid, closes every path, and renders correctly in the browser and in
   Inkscape.

If any of these fail, the failure mode and the chosen workaround should be recorded
here before the feature is considered complete.
