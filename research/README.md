# Random Island Outlines — Research Notes

**Status:** research notes. The recommended generator is now implemented in
`code/island_generator.py` and exposed as the `generate` subcommand in
`code/cli.py`; this folder remains the rationale, parameter reference and
evaluation plan. No application/UI wiring was added.

**Date:** 2026-11-11 · **Scope:** outline generation only · **Target:** the
`chunkMapSeparator` raster-to-SVG pipeline.

**Question:** how do we generate a random, natural-looking island *outline* for a
given width and height, with parameters we can actually control?

---

## TL;DR — the recommended approach

The best cost/quality trade-off is a **noise-displaced signed distance field around
a controlled silhouette**, followed by contour extraction and vector smoothing.

In one sentence: **shape first, erode it with fractal noise, then vectorise the
coastline.**

```
base silhouette  ->  signed distance field  ->  multi-octave noise + domain warp
                 ->  displaced field  ->  sea level / target area
                 ->  mask cleanup  ->  contour  ->  simplify + smooth  ->  SVG/PNG
```

This is a hybrid, not a single algorithm:

| Layer | Chosen technique | Job |
|---|---|---|
| Macro shape | rotated ellipse + 2–5 low-frequency lobes (optional metaball union) | where the island is, how big, how elongated, which way it points |
| Shape representation | signed distance field (SDF) | makes "inside/outside" explicit and gives a clean margin |
| Coast detail | fBm built from Perlin/Simplex/OpenSimplex noise, 4–9 octaves (auto) | fractal roughness at every scale |
| Naturalness | domain warping of the noise lookup | bends the coast into bays and capes instead of a noisy circle |
| Size control | solve the sea level for a target land coverage | "30% of the reference square is land" actually means 30% |
| Cleanup | connected components, small-hole fill, tiny morphology | one coherent island, no pixel speckle |
| Outline | marching squares (or `cv2.findContours`) → RDP → Chaikin / Bézier | the final vector path |

The important finding: **Perlin noise is the detail engine, not the island engine.**
Raw noise thresholded on its own gives land that runs off the canvas, random lakes,
and an arbitrary number of fragments. The distance field (or falloff) is what turns
noise into an island; domain warping is what stops it looking like a noisy circle.

---

## Answers at a glance

| Question | Short answer | Where |
|---|---|---|
| Is Perlin noise enough on its own? | No. It is the detail engine; thresholding it directly does not make a controllable island. | [02 §3.1](02-algorithm-survey.md#31-raw-fbm-threshold) |
| What is the core algorithm? | A controlled base silhouette, converted to a signed distance field, displaced by warped multi-octave fBm, then thresholded and vectorised. | [03](03-recommended-pipeline.md) |
| How is width/height handled? | Work in aspect-corrected reference units; solve coverage against the short-side square; scale the path to the requested canvas. | [03 §1](03-recommended-pipeline.md#1-coordinate-system-and-working-resolution) |
| How are size and shape controlled? | `coverage`, `base_radius`, `aspect`, `rotation`, `lobes`, `lobe_strength`. | [03 §10](03-recommended-pipeline.md#10-parameter-table) |
| How is roughness controlled? | `hurst` (gain), `octaves`, `lacunarity`, `warp`, `coast_amp`, `coast_band`. | [02 §2](02-algorithm-survey.md#2-the-hurst-parameter-and-why-gain-05-is-not-enough), [03 §4](03-recommended-pipeline.md#4-step-3-noise-field) |
| Which image operations matter? | Distance transform, Gaussian blur, remap/domain warp, threshold/quantile, morphology, connected components, contour extraction, RDP, Chaikin. | [03 §9.5](03-recommended-pipeline.md#95-image-operations-in-order) |
| How do we know it is natural? | Fractal dimension, compactness/convexity, diversity, margin/topology checks, plus blind visual review. | [04](04-evaluation-and-integration.md) |

---

## Why this mix

1. **Natural coastlines are level sets of a fractal field.** A coastline is where
   "land height" crosses sea level; it inherits the scaling of the underlying field.
   Multi-octave noise reproduces that scaling, and a 2D field gives a closed contour
   for free.
2. **A pure falloff makes a potato.** A radial/elliptical mask guarantees the island
   stays on the canvas, but it also makes every island round. Low-frequency lobes and
   domain warping break the circle before the fine detail is added.
3. **A pure noise threshold makes a mess.** It has no notion of "island": it can touch
   every edge, produce dozens of components, and change area wildly with the seed.
   A distance field plus a coverage solver fixes all three.
4. **The last 20% is vectorisation.** Pixel-level contours look like stair steps even
   when the underlying shape is good. Subpixel marching squares, RDP and a light
   corner-cutting pass matter as much as the noise.
5. **This project already owns the hard vectoriser.** Generating an alpha mask and
   feeding the existing `code/cli.py` pipeline is the smallest useful integration;
   generating the outline directly is the faster preview path.

---

## Recommended defaults

Starting point for a 1024 px working grid; all lengths are fractions of the short
side unless marked `px`. Full definitions and ranges are in
[03-recommended-pipeline.md](03-recommended-pipeline.md).

| Parameter | Default | What it does |
|---|---:|---|
| `coverage` | `0.30` | land fraction of the short-side square |
| `base_radius` | `0.30` | base island radius, as a fraction of the short side |
| `margin` | `0.08` | water gap kept between coast and canvas edge |
| `aspect` | `1.2` | long axis / short axis of the base silhouette |
| `rotation` | random | orientation of the long axis |
| `lobes` | `5` | number of low-frequency lobe harmonics (approximate) |
| `lobe_strength` | `0.30` | how far the silhouette departs from an ellipse |
| `hurst` | `0.74` | fBm roughness; higher = smoother (`D ≈ 2 - H`) |
| `octaves` | auto (4–7) | number of noise scales |
| `lacunarity` | `2.0` | frequency ratio between octaves |
| `warp` | `0.32` | domain-warp displacement, in island radii |
| `coast_amp` | `0.18` | coast displacement, in island radii |
| `detail_px` | `2.0` | smallest feature worth generating |
| `smooth` | `1` | Chaikin iterations after simplification |
| `simplify_px` | `0.6` | RDP tolerance |
| `islands` | `1` | number of separate landmasses |
| `lagoons` | `false` | keep large enclosed water bodies |

`coverage` is measured against the **short-side square**, not the full canvas:
a 0.30 island occupies 30% of the largest centred square that fits the canvas, so
its physical size stays consistent on 16:9 and 9:16 canvases. See
[03-recommended-pipeline.md](03-recommended-pipeline.md#51-solving-for-coverage)
for the alternative (fraction of the full canvas) and why it makes wide canvases
much harder.

`hurst = 0.74` with `lacunarity = 2` means an amplitude gain of
`2^-0.74 ≈ 0.60`, not the usual `0.5`. This is deliberate: the common
`gain = 0.5` fBm has `H = 1`, which is visually smooth and gives coastlines that
look soft and "melted" rather than fractal. See
[02-algorithm-survey.md](02-algorithm-survey.md#2-the-hurst-parameter-and-why-gain-05-is-not-enough).

---

## What the research found

- **Raw Perlin/fBm thresholding is not enough.** It is a good detail generator but a
  poor island generator: no size control, no margin, no component control.
- **The standard fBm gain of 0.5 is smoother than most natural coasts.** A gain of
  about `0.60` (`H ≈ 0.74`) gives a coastline dimension near `1.26`, which is in the
  range usually quoted for real coasts. The gain is the single most important
  "ruggedness" knob.
- **Domain warping is the biggest naturalness win after the falloff.** It turns
  concentric noise rings into bays, capes and asymmetric lobes, without adding a new
  algorithm.
- **Distance fields give better control than a plain radial falloff.** A signed
  distance field makes it easy to keep the island away from the border, union
  several islands, subtract a lagoon, and limit how far the coast can wander.
- **Do not smooth the field; smooth the contour.** Strong Gaussian blur or large
  morphology destroys exactly the small bays that make a coast look natural. A light
  contour-level smoothing keeps the silhouette and removes pixel stair steps.
- **The existing vectoriser is a good fit.** `code/cli.py` already turns an alpha mask
  into smooth cubic Béziers. Generating a clean mask and letting it do the fitting is
  less work and better tested than writing a second Bézier fitter.

---

## Method (and what is still unproven)

The comparison is based on the classic noise/fractal papers, Red Blob Games'
terrain and distance-field notes, Chema Guerra's island series, modern island
tutorials, and the OpenCV/scikit-image documentation. The methods were organised by
the three layers (macro shape, coast texture, outline) and checked against this
repo's alpha-mask → contour → Bézier constraints.

The parameter values in this folder are **starting points**, not measured optima.
They come from the geometry of the problem (coverage, margin, fractal dimension) and
from the ranges used in the cited sources. The prototype in
[04-evaluation-and-integration.md](04-evaluation-and-integration.md) is what should
confirm or adjust them.

---

## What is in this folder

| File | Contents |
|---|---|
| [01-problem-and-naturalness.md](01-problem-and-naturalness.md) | requirements, what "natural" means in measurable terms, constraints from this repo |
| [02-algorithm-survey.md](02-algorithm-survey.md) | Perlin/Simplex/fBm, falloffs, warping, Voronoi, DLA, cellular automata, reaction–diffusion, midpoint displacement; comparison matrix and verdicts |
| [03-recommended-pipeline.md](03-recommended-pipeline.md) | the recommended pipeline step by step, formulas, pseudocode, parameter table, presets, failure modes, integration notes |
| [04-evaluation-and-integration.md](04-evaluation-and-integration.md) | acceptance criteria, metrics, test matrix, how to score results with the existing accuracy tool |
| [05-references.md](05-references.md) | annotated sources, grouped by topic |
| [assets/pipeline.svg](assets/pipeline.svg) | the recommended pipeline as a diagram |
| [assets/island-anatomy.svg](assets/island-anatomy.svg) | the shape features the parameters are meant to control |

---

## What was deliberately left out

- No editor/UI, no web endpoint, and no application integration. The generator is
  available from `code/cli.py generate`.
- No new package in `code/requirements.txt`. The implementation uses `numpy`,
  `opencv-python` and `scipy`, all already present. `opensimplex` or a
  FastNoiseLite binding remains an optional quality upgrade.
- No colour/style work beyond the CLI's simple land colour. This is about the
  outline/mask only.
- No archipelagos or lagoons in the first implementation; those are documented as
  extensions, not shipped.

## Implementation notes (added after the research)

The shipped defaults are slightly stronger than the first estimates in this folder
because visual testing showed that the initial values made circular, fuzzy-edged
blobs. The defaults now used by `code/island_generator.py` are:

- `lobes = 5`, `lobe_strength = 0.30`
- `warp = 0.32`
- `coast_amp = 0.18`, `coast_band = 0.15`
- Perlin fBm at 4–7 octaves (capped for batch speed)
- one island by default; `--islands N` for several placements
- default output is a transparent PNG with a constant-width outline; `--filled`
  restores the solid land colour
- optional `--subdivisions N`: relaxed, area-aware regions per island, with tiny
  cells merged into neighbours and slightly warped boundaries
- holes are filled; lagoons are still deferred

The earlier parameter tables have been updated to match. The underlying recipe is
unchanged: silhouette → SDF → warped fBm → coverage solve → cleanup → PNG.

Current performance is about 0.5 s per 512 px island, 1.5–2.5 s per 1024 px island,
and a little more with subdivisions on the development machine. That is above the
aspirational interactive targets in
[04-evaluation-and-integration.md](04-evaluation-and-integration.md); lowering the
working resolution, caching the noise fields, or moving the noise to a compiled
library are the obvious next optimisations.

---

## Decisions still open / deferred

The first implementation answers some of these; the rest are deliberately deferred.

1. **Filled SVG vs outline SVG.** The generator writes an outline PNG by default
   (or a filled alpha mask with `--filled`). How the SVG writer represents
   holes/lagoons is still open (even-odd fill or separate paths).
2. **Archipelagos.** v1 generates one landmass per PNG. The SDF union extension is
   documented but not implemented.
3. **Lagoons and inland water.** v1 fills all holes. Keeping them needs the SVG
   fill-rule work and a hole-area policy.
4. **Realistic vs cartographic.** v1 has `natural`, `smooth` and `rugged` presets;
   the exact style target is still a product decision.
5. **Direct SVG.** v1 goes raster → existing CLI. Direct contour output is the next
   quality step if the round trip becomes a bottleneck.
6. **Dependencies.** v1 uses only NumPy, OpenCV and SciPy. A Simplex/OpenSimplex
   implementation remains an optional quality upgrade.
7. **Performance.** Generation is roughly 1–2 s per 1024 px island on the
   development machine; interactive previews would need a smaller grid or caching.
