# 02 — Algorithm Survey

This document compares the candidate techniques. It is deliberately broader than
the recommendation: some methods are included because they are popular, some because
they solve a specific sub-problem well, and some because they are commonly suggested
but are a poor fit for this project.

The problem has three separable layers, and it helps to judge every method by which
layer it addresses:

1. **Macro shape** — where the island is, how big it is, its elongation, lobes and
   orientation.
2. **Coast texture** — the multi-scale roughness along the boundary.
3. **Outline extraction** — turning the shape into a clean, closed vector path.

No single classic algorithm does all three well. The recommendation is a
composition; the survey explains why each piece was chosen.

---

## 1. Noise primitives

### 1.1 Value noise

Random values on a lattice, interpolated with a smoothstep/bicubic kernel. It is
the cheapest coherent noise to implement and is easy to vectorise in NumPy.

- **Strengths:** trivial, fast, deterministic, no dependency.
- **Weaknesses:** blocky, axis-aligned artefacts, visible square structure unless
  rotated and warped; lower quality than gradient noise.
- **Role:** acceptable for an MVP if every octave is rotated and domain-warped;
  not the best choice for the final look.

### 1.2 Perlin noise

Gradient noise introduced by Ken Perlin in 1983 and published in 1985
([Perlin, 1985](https://doi.org/10.1145/325165.325247);
[overview](https://en.wikipedia.org/wiki/Perlin_noise)). Each lattice corner has a
random gradient; the value is the dot product of the gradient with the offset,
interpolated across the cell using a smooth kernel.

- **Strengths:** fast, well understood, available everywhere, good "natural texture"
  character, no patent issues.
- **Weaknesses:** axis-aligned artefacts (especially visible when thresholded),
  values are zero at lattice points, and the classic 2D implementation has a
  limited set of gradient directions. Octaves sampled at integer multiples of the
  same coordinates are correlated.
- **Role:** a perfectly good fBm building block if octaves are rotated/offset and
  the result is domain-warped. Simplex/OpenSimplex are better if available.

### 1.3 Simplex noise and OpenSimplex

Simplex noise (Perlin, 2001) uses a simplicial grid instead of a square lattice. It
has fewer directional artefacts, lower complexity in higher dimensions, and a
well-defined continuous gradient
([overview](https://en.wikipedia.org/wiki/Simplex_noise)). The original patent
covered 3D+ texture synthesis and expired on 8 January 2022, so the licensing
concern that motivated OpenSimplex is now historical.

OpenSimplex (Kurt Spencer) is a patent-free alternative with similar visual
quality; the Python package exposes `noise2array` and is NumPy-friendly
([PyPI](https://pypi.org/project/opensimplex/)). FastNoiseLite is a portable,
MIT-licensed library with many noise types and is a good C/C++/C#/JS/Rust choice
([GitHub](https://github.com/Auburn/FastNoiseLite)).

- **Strengths:** fewer grid artefacts, better isotropy, good for thresholded
  contours.
- **Weaknesses:** more code than value noise; Python implementations can be slow
  unless vectorised.
- **Role:** the preferred noise primitive for the final generator.

### 1.4 Worley / cellular noise

Distance to the nearest of a set of feature points. Produces cells, cracks and
"rocky" patterns.

- **Strengths:** excellent for archipelagos, rock outcrops, coral, and cellular
  detail; naturally produces separated blobs.
- **Weaknesses:** not a smooth heightfield; thresholded cells look polygonal; hard to
  make a single coherent coastline.
- **Role:** a secondary texture for offshore islets or rocky detail, not the primary
  island shape.

### 1.5 Diamond–square / midpoint displacement

Recursive midpoint displacement on a square grid, introduced by Fournier, Fussell
and Carpenter (1982)
([overview](https://en.wikipedia.org/wiki/Diamond-square_algorithm)). It is simple
and produces fractal terrain, but the classic version has visible horizontal and
vertical creases because the largest perturbation happens on a rectangular grid;
Miller (1986) and Lewis (1987) documented the artefacts and fixes.

- **Strengths:** very little code, fractal, good for a quick demo.
- **Weaknesses:** grid creases, power-of-two sizing, poor control of the macro
  shape, and the 1D/closed-loop variant can self-intersect if displacement is
  strong.
- **Role:** not recommended for the main generator. The closed-loop variant is a
  reasonable fast fallback for an outline-only mode (see §3.11).

### 1.6 Spectral synthesis (FFT)

Generate white noise in the frequency domain, multiply by a power-law filter, and
inverse-FFT. The Hurst exponent is controlled directly by the spectral exponent, and
the result can be made tileable.

- **Strengths:** statistically clean, precise `H`, fast for large grids, tileable,
  easy to make anisotropic (stretch the frequency axes).
- **Weaknesses:** global and non-local; hard to control a specific silhouette;
  requires an FFT and careful normalisation; thresholded results still need a mask
  or falloff.
- **Role:** an advanced heightfield generator; a good alternative to octave summing
  when statistical correctness or tiling matters.

### 1.7 Ridged and billow variants

`ridged = 1 - |noise|` produces sharp ridges; `billow = |noise|` produces puffy
clouds. Both are useful for mountains and clouds.

- **Role:** interior mountain detail or stylised texture. Not a coastline primitive:
  ridged noise creates sharp creases, which read as mountain ridges rather than
  coast.

### 1.8 Summary

| Primitive | Natural coast quality | Control | Speed | Complexity | Best role |
|---|---|---|---|---|---|
| Value noise | fair | fair | very fast | very low | MVP / fallback |
| Perlin | good | good | fast | low | fBm detail |
| Simplex / OpenSimplex | very good | good | fast | medium | preferred detail |
| Worley | poor alone | fair | medium | medium | islets, rocks |
| Diamond–square | fair | poor | fast | low | demo, not final |
| Spectral FFT | very good | fair | fast (large) | high | advanced heightfield |
| Ridged/billow | poor for coast | fair | fast | low | mountains/clouds |

---

## 2. The Hurst parameter, and why gain 0.5 is not enough

This is the most useful quantitative finding in the research, and it is easy to get
wrong.

A standard fBm is a sum of noise octaves:

```
F(p) = sum_{i=0..O-1} g^i * noise(l^i * p + offset_i) / sum(g^i)
```

where `l` is the lacunarity (frequency multiplier, usually 2) and `g` is the gain
(amplitude multiplier, often 0.5). The Hurst exponent of the resulting field is

```
H = log(1/g) / log(l)
```

For the common `l = 2, g = 0.5`, this gives `H = 1`. An fBm with `H = 1` is smooth;
its graph has dimension `2 - H = 1`, i.e. it is not fractal at all in the usual
sense. It still looks like terrain because the lowest octave dominates and because
thresholding creates a coastline, but the coastline itself is soft.

For a 2D fBm field with Hurst `H`, a level set (the coastline) has dimension
approximately `2 - H`. This is the number to compare with the `1.1–1.4` range for
natural coasts. Treat it as a first-order design guide, not an exact promise:
finite octave counts, the noise kernel and thresholding all change the effective
dimension.

| Gain `g` (lacunarity 2) | Hurst `H` | Approx. coast dimension `2-H` | Character |
|---:|---:|---:|---|
| 0.50 | 1.00 | 1.00 | very smooth, "melted" |
| 0.55 | 0.86 | 1.14 | smooth, cartographic |
| 0.60 | 0.74 | 1.26 | natural, recommended |
| 0.65 | 0.62 | 1.38 | rugged |
| 0.70 | 0.51 | 1.49 | fjord-like / very rough |
| 0.75 | 0.42 | 1.58 | approaching coral/DLA |

So the usual game-dev advice "gain 0.5" is not wrong for rolling terrain, but it is
too smooth for a natural coastline. **Gain ≈ 0.60 is the recommended default** for
this project, with 0.65 for a rugged preset and 0.55 for a smooth/cartographic
preset. The trade-off is real: higher gain adds high-frequency energy everywhere, so
the coastal-band modulation in
[03-recommended-pipeline.md](03-recommended-pipeline.md) becomes more important.

Source for the fBm definition and the `2 - H` graph dimension:
[Fractional Brownian motion](https://en.wikipedia.org/wiki/Fractional_Brownian_motion).
The coastline interpretation follows Mandelbrot's coastline work
([Mandelbrot, 1967](https://doi.org/10.1126/science.156.3775.636);
[coastline paradox](https://en.wikipedia.org/wiki/Coastline_paradox)).

---

## 3. Shape-generation families

### 3.1 Raw fBm + threshold

Generate a 2D fBm field over the canvas, pick a sea level, mark `field >= level` as
land.

**What it gives:** a natural-looking texture, and a closed contour for free. This is
the "simplest thing that could possibly work" and is the basis of many tutorials
([Red Blob Games, 2015](https://www.redblobgames.com/maps/terrain-from-noise/)).

**Why it fails as an island generator:**

- land frequently touches the canvas border;
- the number of connected components is arbitrary;
- the land fraction varies wildly with the seed;
- there is no control over position, elongation or orientation;
- holes/lakes appear wherever the noise dips.

**Verdict:** useful as a detail layer, not as the shape layer.

### 3.2 fBm + radial/elliptical falloff

Amit Patel's "two ingredients" formulation is the classic fix
([Red Blob Games, 2015](https://www.redblobgames.com/maps/terrain-from-noise/#islands)):

1. a **distance** function `d` that is 0 at the centre and 1 at the border;
2. a **shaping** function that mixes the noise with `1 - d`, e.g.
   `e = lerp(e, 1 - d, mix)`.

Common distance functions are the Euclidean distance, the squared Euclidean distance,
or the "square bump" `1 - (1 - nx^2)(1 - ny^2)`, which fills a rectangular canvas
more completely. The result is guaranteed to be water at the border, and the `mix`
parameter controls how much of the original noise survives.

**Strengths:** tiny amount of code, robust, easy to understand, guaranteed edge
water.

**Weaknesses:**

- the macro shape is a circle/ellipse (a "potato");
- the transition from land to water is a broad gradient, so the coastline is often
  soft and low-detail;
- the `mix` parameter trades off shape control against naturalness;
- it is hard to make deep bays or an elongated island without extra machinery.

**Verdict:** a good baseline and a good fallback, but not the final shape engine.
The recommended pipeline keeps the *idea* of a falloff but replaces the scalar
falloff with a signed distance field and adds low-frequency asymmetry and warping.

### 3.3 Warped radial function / Fourier harmonics

Represent the outline directly in polar coordinates:

```
r(theta) = R * (1 + sum_k a_k * cos(k*theta + phi_k) + noise(theta))
```

The `a_k` terms give lobes; the noise term gives detail. Because `r` is a
single-valued function of `theta`, the curve is star-shaped around the centre and
cannot self-intersect (as long as `r > 0`).

**Strengths:** extremely fast; guaranteed simple closed curve; direct control of
area, elongation and lobe count; no raster/contour step; easy to make periodic.

**Weaknesses:** star-shaped only. A deep bay that would "wrap around" the centre
cannot be represented; the island always has a line of sight to the centre. The
result tends to look like a spiky potato unless the low-frequency harmonics are
carefully chosen.

**Verdict:** excellent for small islands, icons and a fast preview. Not sufficient
for a natural large island with bays and peninsulas.

### 3.4 Signed distance field + noise displacement (recommended)

Build a controlled base silhouette (ellipse + lobes, or a union of metaballs),
convert it to a signed distance field `phi` (positive inside, negative outside), and
displace the field with noise:

```
F(x, y) = phi(x, y) + A * N(x, y)
land = F >= sea_level
```

The coastline is the zero (or sea-level) level set of `F`. Because `phi` grows
inland, the interior stays land; because `phi` falls outside, the exterior stays
water. The noise changes *where* the zero crossing is, not the topology of the base
shape (unless the displacement is large enough to close a bay or split a neck, which
can be desirable).

**Strengths:**

- separates macro shape (phi) from coast texture (N) cleanly;
- direct control of margin, position, elongation and lobe count;
- union of several base shapes gives archipelagos; subtraction gives lagoons;
- works with a raster distance transform, so arbitrary base shapes are allowed;
- the displacement amplitude gives a direct "how far can the coast wander"
  parameter;
- a single scalar field can later carry interior elevation for biomes.

**Weaknesses:**

- requires a base-shape design step (more parameters than a plain falloff);
- a raster distance transform adds a pass (cheap with OpenCV);
- strong displacement can merge/split components, so cleanup is still needed;
- if the base shape is a plain ellipse, the result is still an ellipse with a rough
  edge; the lobes/warp are what make it natural.

**Verdict:** the best balance for this project. This is the core of the
recommendation.

### 3.5 Metaballs / Gaussian blobs

Sum a set of smooth radial influence functions and threshold the result:

```
B(x) = sum_i w_i * exp(-0.5 * (x - c_i)^T Sigma_i^-1 (x - c_i))
land = B >= threshold
```

Blobs that are close together merge into a single organic shape; anisotropic
covariance makes elongated lobes; negative blobs carve bays
([Blinn, 1982](https://doi.org/10.1145/357306.357310);
[overview](https://en.wikipedia.org/wiki/Metaballs)).

**Strengths:** intuitive control ("put a lobe here"), smooth merges, naturally
supports archipelagos and ring/atoll shapes; no raster pass required if the
potential is evaluated analytically.

**Weaknesses:** the result is smooth and blobby; it needs noise displacement to look
like a coast; placing blobs to get a pleasing silhouette is itself a small design
problem; merging can produce unexpected bridges.

**Verdict:** a strong **base-shape generator** for the SDF pipeline, especially for
archipelagos and lagoons. Not a complete solution on its own.

### 3.6 Voronoi / Delaunay polygon mesh

Scatter points with Poisson-disk sampling, build a Delaunay triangulation and its
Voronoi dual, assign elevation to cells/vertices, and extract the coast along cell
boundaries. This is the approach behind Red Blob Games' polygon map generation and
Chema Guerra's island series
([Red Blob polygon maps](http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/);
[Guerra, 2025, part I](https://brashandplucky.com/2025/09/07/procedural-island-generation-i.html)).

**Strengths:** irregular, organic tessellation; excellent control points for
mountains, rivers and biomes; a stylised low-poly look is easy; coastlines follow
cell edges and can be smoothed.

**Weaknesses:** much more machinery (Poisson disk, Delaunay, Voronoi, graph
traversal); the raw coast is angular; getting a smooth natural outline requires
extra smoothing; overkill for an outline-only feature.

**Verdict:** the right direction if this becomes a full map generator with biomes
and rivers. Too heavy for the first island-outline version.

### 3.7 Cellular automata

Start with random land/water cells and repeatedly apply a smoothing rule (e.g. a
cell becomes land if enough neighbours are land). This produces cave-like blobs and
lagoons.

**Strengths:** very simple; good for caves, lagoons and "swiss cheese" islands;
naturally discrete.

**Weaknesses:** scale is tied to cell size; results are either blobby or noisy;
grid artefacts; no direct control of coastline fractal dimension; hard to get a
clean, natural silhouette without a lot of post-processing.

**Verdict:** a style variant, not the primary generator.

### 3.8 Diffusion-limited aggregation (DLA)

Random walkers stick to a growing cluster. In 2D the resulting cluster has a fractal
dimension of about 1.71
([Witten & Sander, 1981](https://doi.org/10.1103/PhysRevLett.47.1400);
[overview](https://en.wikipedia.org/wiki/Diffusion-limited_aggregation)).

**Strengths:** beautiful, intricate, coral-like branching; excellent for tiny
islets, reefs or decorative detail.

**Weaknesses:** far too branched and porous for a solid island; slow; area and
silhouette are hard to control; the result is not a closed simple region without
filling/smoothing.

**Verdict:** not suitable as the primary shape. Could be a detail generator for
offshore rocks in a stylised map.

### 3.9 Reaction–diffusion

Two or more chemicals diffuse and react, producing spots, stripes, labyrinths and
other Turing patterns
([Turing, 1952](https://doi.org/10.1098/rstb.1952.0012);
[overview](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)).
The Gray–Scott model is the common computer-graphics demo.

**Strengths:** organic, unusual patterns; good for lagoons, reefs and alien terrain.

**Weaknesses:** parameter-sensitive; pattern scale is not directly controllable;
requires many iterations (slow); the result is a texture, not a controlled island
shape.

**Verdict:** interesting for a stylised variant; not a practical first
implementation.

### 3.10 Space colonisation

Branches grow toward nearby attraction points, producing tree-like structures
([Runions et al., 2007](https://www.algorithmicbotany.org/papers/colonization.egwnp2007.html)).

**Strengths:** excellent for rivers, deltas, roots and dendritic peninsulas.

**Weaknesses:** it generates a branching skeleton, not a solid closed island; turning
it into a coastline requires a hull or a distance-field thickening step.

**Verdict:** useful later for river networks or a delta-shaped island, not for the
base outline.

### 3.11 Midpoint displacement on a closed contour

Take a closed base polygon (ellipse or a hand-authored silhouette), recursively
subdivide each edge, and displace the new midpoint along the edge normal by a random
amount scaled by `edge_length^H`. This is the 1D/closed analogue of diamond–square.

**Strengths:** direct outline; no raster step; fractal at all scales; very fast;
preserves the base silhouette if displacement is moderate; easy to keep periodic and
closed.

**Weaknesses:** the base shape still dominates; strong displacement self-intersects;
no holes or multiple islands without extra work; controlling the large-scale bays is
hard because the base polygon has to contain them.

**Verdict:** a good **fast fallback** for outline-only mode, and a good way to add
final detail to a smoothed contour. Not the primary method if interior terrain or
topology control matters.

### 3.12 Spectral synthesis as a shape source

Use FFT-filtered noise as the heightfield, then apply a falloff or SDF mask. This
combines the statistical quality of spectral noise with the control of a mask.

**Verdict:** a quality upgrade for the noise layer, not a replacement for the shape
layer. Worth prototyping if octave summing shows banding or repetition.

---

## 4. Hybrid patterns that work

### 4.1 Base silhouette + noise-displaced SDF (primary)

The recommended composition:

```
base shape -> SDF -> + fBm (warped) -> threshold -> cleanup -> contour
```

It combines the control of a base shape, the naturalness of fBm, and the topology
guarantees of a level set. The base shape can be an ellipse with lobes, a metaball
union, or a hand-drawn polygon; the rest of the pipeline does not change.

### 4.2 Falloff + domain warp + coastal band (simpler)

If the extra SDF machinery is unwanted, a plain elliptical falloff plus strong
domain warping and a coastal detail band gets most of the way there:

```
d = elliptical distance from centre
falloff = smoothstep(1, 0, d)
e = falloff * (1 - k) + k * fbm(warp(p))
```

It is smaller and faster, but it cannot represent deep bays or subtract a lagoon,
and the macro shape remains fundamentally elliptical. Good for a first prototype.

### 4.3 Multi-level threshold

Threshold the field at several levels to get the main island, small offshore islets,
and shallow shelves. Keep components by area. This is a cheap way to add realism
without a second generator.

### 4.4 SDF union and subtraction

- **Union** of several base shapes: `phi = min(phi_i)` → archipelago or a compound
  island.
- **Subtraction**: `phi = max(phi_base, -phi_hole)` → lagoon, bay or channel.
- **Smooth union**: a soft-min of the distances → natural bridges between lobes.

This is the cleanest way to support archipelagos and lagoons later.

### 4.5 Voronoi + noise (stylised)

Use Voronoi cells for the macro shape and noise for the coast; smooth the cell edges.
This gives a low-poly/stylised look rather than a realistic one. It is a good
alternative preset if the product wants a "hand-drawn map" style.

### 4.6 Radial harmonics + midpoint displacement (fast preview)

Generate a star-shaped outline from harmonics, then add fractal displacement. This is
the fastest path to a reasonable-looking island and is useful for interactive
previews or icons. It is not a substitute for the full pipeline when bays, holes or
archipelagos are required.

---

## 5. Comparison matrix

Scores are relative to this project, not absolute.

| Method | Macro control | Coast naturalness | Speed | Implementation effort | Holes/lagoons | Archipelagos | Fits raster→SVG pipeline |
|---|---|---|---|---|---|---|---|
| Raw fBm + threshold | none | medium | fast | low | accidental | accidental | yes |
| fBm + radial falloff | medium | medium | fast | low | rare | no | yes |
| Radial harmonics | high (star-shaped) | low–medium | very fast | low | no | no | yes (direct) |
| **SDF + noise** | **high** | **high** | **medium** | **medium** | **yes** | **yes** | **yes** |
| Metaballs | high | low alone | fast | low–medium | yes | yes | yes |
| Voronoi/Delaunay | high | medium (stylised) | slow | high | yes | yes | yes |
| Cellular automata | low | low–medium | medium | low | yes | yes | yes |
| DLA | low | high (too fractal) | slow | medium | no | yes | yes |
| Reaction–diffusion | low | medium (stylised) | slow | medium | yes | yes | yes |
| Space colonisation | low | low (branching) | medium | medium | no | no | via hull |
| Midpoint displacement | medium | medium–high | very fast | low | no | no | yes (direct) |
| Spectral FFT | medium | high | fast (large) | high | via threshold | via threshold | yes |

"Fits raster→SVG pipeline" means the method can produce a binary/alpha mask that
`code/cli.py` can consume. Most can; the direct-outline methods can also bypass the
raster step.

---

## 6. Verdict

**Primary recommendation:** SDF + multi-octave warped noise around a controlled
silhouette (§3.4), with an ellipse + lobes base shape for v1 and metaball unions as
the archipelago/lagoon extension.

**Why not the simpler falloff?** It is a good baseline, but it cannot represent deep
bays, does not give a clean margin control, and makes every island fundamentally
elliptical. The SDF version is only a little more code and is much more controllable.

**Why not Voronoi?** It is the right choice for a full terrain/biome generator, but
it is a lot of machinery for an outline. If the product later grows into a map
generator, Voronoi can replace the base-shape layer while keeping the noise,
thresholding and contour stages.

**Why not DLA/reaction–diffusion/cellular automata?** They produce interesting
textures, but they are hard to control, slow to tune, and their characteristic look
is either too fractal (DLA) or too blobby/cave-like (CA, RD) for a natural island
outline.

**Fast fallback:** radial harmonics + midpoint displacement for previews, icons, or
a minimal first version with no interior terrain.

**Stylised alternative:** Voronoi cells + smoothed edges for a low-poly/hand-drawn
map style.

---

## 7. What still needs measuring

The survey narrows the choice; it does not prove the parameters. The first
implementation is in `code/island_generator.py`; a controlled prototype should still
compare:

1. **Noise primitive:** Perlin vs OpenSimplex vs value noise, at the same `H`.
2. **Base shape:** ellipse + lobes vs metaball union.
3. **Coast modulation:** global noise vs coastal-band noise.
4. **Contour extraction:** `cv2.findContours` vs marching squares.
5. **Smoothing:** none vs Chaikin vs cyclic Gaussian vs direct Bézier fit.

The evaluation plan in
[04-evaluation-and-integration.md](04-evaluation-and-integration.md) lists the
metrics and the test matrix for that comparison.
