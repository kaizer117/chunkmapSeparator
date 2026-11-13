# 03 — Recommended Pipeline

This is the concrete proposal. It is written so it can be implemented later without
re-deciding the architecture, but it contains **no implementation code**. The
pseudocode is explanatory.

The pipeline is:

```
parameters + seed
      |
      v
base silhouette  (rotated ellipse + low-frequency lobes, optional metaballs)
      |
      v
signed distance field  phi  (positive inside, negative outside)
      |
      v
multi-octave noise N  (fBm, domain-warped, coastal band)
      |
      v
displaced field  F = phi + A * N
      |
      v
sea level t  (solved for target coverage)  +  edge safety
      |
      v
binary mask  ->  cleanup  (components, holes, tiny morphology)
      |
      v
contour  (marching squares, or cv2.findContours)
      |
      v
simplify  (RDP)  ->  smooth  (Chaikin / cyclic Gaussian / Beziers)
      |
      v
SVG path(s) and/or RGBA mask
```

The reason for the two-stage shape (base silhouette, then noise displacement) is
explained in [02-algorithm-survey.md](02-algorithm-survey.md). In short: the base
silhouette controls *where* the island is; the noise controls *what the coast looks
like*; the distance field is the contract between them.

---

## 1. Coordinate system and working resolution

Work in a grid that preserves the aspect ratio of the requested canvas:

```
S = 1024                         # working long side, default
Gw = round(S * width / max(width, height))
Gh = round(S * height / max(width, height))
```

`Gw`/`Gh` are the working grid; the final SVG/mask is scaled to `width`/`height`.
For a 16:9 canvas this gives `1024 x 576`; for 9:16, `576 x 1024`.

Use **reference pixels** for geometry:

```
s = min(Gw, Gh)                  # short side in pixels
x = (col + 0.5) / s              # 0 .. Gw/s
y = (row + 0.5) / s              # 0 .. Gh/s
```

So the short side is `1.0` unit long, and both axes have the same physical scale.
This is important: normalising x and y separately by `Gw`/`Gh` would stretch the
island on non-square canvases.

All lengths below are in these units unless marked `px`:

- `margin` and `base_radius` are fractions of the short side.
- `coast_amp` and `warp` are in **island radii**; multiply by `base_radius` to
  convert them to short-side units.
- `detail_px` is in output pixels; convert to working pixels by
  `detail_work = detail_px * s / min(width, height)`.

For very large canvases, generating at `S = 1024` and scaling the path is usually
enough; a working pixel becomes about 4 output pixels at 4096 px, which is fine for
a coastline. If the output needs genuine 4K detail, raise `S` to 2048 or generate a
second, higher-frequency detail pass.

---

## 2. Step 1 — base silhouette

### 2.1 Option A: rotated ellipse + radial lobes (recommended for v1)

Place the island centre `(cx, cy)` near the canvas centre, with a small seeded
jitter:

```
cx = Wc/2 + jitter_x,   cy = Hc/2 + jitter_y
jitter_x, jitter_y ~ Uniform(-center_jitter, +center_jitter) * s
```

where `Wc = Gw/s`, `Hc = Gh/s`.

The base radius preserves area under elongation:

```
rx = base_radius * sqrt(aspect)
ry = base_radius / sqrt(aspect)
```

Rotate the sample point by `-rotation` around the centre, then compute the
normalised elliptical radius and angle:

```
dx = (x - cx) * cos(rot) + (y - cy) * sin(rot)
dy = -(x - cx) * sin(rot) + (y - cy) * cos(rot)
rho = sqrt((dx/rx)^2 + (dy/ry)^2)
theta = atan2(dy/ry, dx/rx)
```

Add low-frequency lobes. A small Fourier series is enough:

```
R(theta) = 1 + sum_{k=2..K} a_k * cos(k*theta + phi_k)
a_k ~ Normal(0, lobe_strength / k)
phi_k ~ Uniform(0, 2*pi)
K = max(3, lobes)
```

Clamp `R` to `[0.55, 1.45]` so the shape stays a simple star-shaped region and does
not collapse. The base mask is:

```
base = rho <= R(theta)
```

Why `a_k` decays with `k`: the low harmonics create large lobes and bays; the high
harmonics create small scallops and would otherwise compete with the noise. A
`1/k` decay is a simple, controllable choice. `lobe_strength = 0.25–0.30` gives
a clearly non-circular silhouette without producing a star shape.

`aspect > 1` makes the island longer along the rotated x axis; `aspect < 1` makes it
longer along y. `rotation` is in radians internally, degrees at the API boundary.

### 2.2 Option B: union of metaballs (for archipelagos and organic lobes)

Define 2–5 anisotropic Gaussian blobs along a random spine:

```
B(x, y) = sum_i w_i * exp(-0.5 * d_i^T Sigma_i^-1 d_i)
base = B >= 1
```

with `d_i = (x - c_i)` rotated into the blob's frame, `Sigma_i` diagonal with
`sigma_x`, `sigma_y`, and `w_i` chosen so each blob reaches the threshold near its
intended radius. Blobs placed within about `2 * sigma` of each other merge into a
single landmass; blobs further apart become separate islands.

For circular blobs, the analytic union distance is approximately:

```
phi_union = min_i (|d_i| - r_i)
```

For anisotropic blobs, use the Mahalanobis distance
`sqrt(d_i^T Sigma_i^-1 d_i)` instead of `|d_i|`. Or use a smooth-min if softer
bridges are wanted. This is optional for v1; the ellipse + lobes shape is simpler
and already gives good results. Metaballs become useful when the product needs
archipelagos, atolls or a "hand-placed lobes" feel.

### 2.3 Option C: hand-authored base polygon

Any closed polygon can be rasterised and used as the base shape. This is useful for
tests, for "generate around this sketch" features, and for reproducing a specific
silhouette. The rest of the pipeline is unchanged. The base polygon must be simple
(no self-intersections); validate or fix before rasterising.

---

## 3. Step 2 — signed distance field

Rasterise the base mask at `Gw x Gh` (uint8, 255 inside, 0 outside), then:

```
din  = cv2.distanceTransform(mask,       cv2.DIST_L2, 5)
dout = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
phi_px = din - dout
phi = phi_px / s
```

`phi` is positive inside, negative outside, and approximately the Euclidean
distance to the base boundary in short-side units. Smooth it lightly to remove
raster quantisation:

```
phi = gaussian_filter(phi, sigma=0.7)   # working pixels
```

`cv2.DIST_L2` with a 5x5 mask is fast and accurate enough; `cv2.DIST_MASK_PRECISE`
is exact but slower. At 1024 px, the difference is invisible after smoothing.

Why a distance field instead of the raw mask:

- The zero crossing is the coast; noise can move it locally without changing the
  mask's topology.
- The magnitude gives a natural "how far inland/offshore" coordinate, which is used
  to modulate detail and to keep the interior clean.
- Union/subtraction of shapes is a `min`/`max` on distances, which makes
  archipelagos and lagoons easy later.
- The margin is explicit: the base shape can be inset by `margin`, and the noise
  amplitude is bounded.

If an analytic base shape is used (ellipse, metaball), `phi` can be computed
directly and no raster distance transform is needed. The raster path is recommended
for v1 because it supports arbitrary base shapes and is only a few lines with
OpenCV.

---

## 4. Step 3 — noise field

### 4.1 fBm with a controlled Hurst exponent

Use a gradient noise `noise2D` (Perlin, Simplex, OpenSimplex, or value noise) and sum
octaves:

```
f_i = base_frequency * lacunarity^i
g   = lacunarity^(-hurst)
N(p) = sum_{i=0..O-1} g^i * noise2D(f_i * R_i(p) + offset_i) / sum(g^i)
```

- `R_i` is a seeded rotation of the sample point. Rotating each octave breaks the
  axis alignment of Perlin/value noise and prevents all octaves from sharing the
  same grid directions.
- `offset_i` is a seeded 2D offset. Without it, octaves sampled at integer
  multiples of the same coordinates are correlated
  ([Red Blob Games](https://www.redblobgames.com/maps/terrain-from-noise/#elevation)).
- `g = lacunarity^(-hurst)` follows directly from the definition of fBm; with
  `lacunarity = 2` and `hurst = 0.74`, `g ≈ 0.599`.
- The division by `sum(g^i)` keeps the range stable as `O` changes.

Choose `O` from the requested detail size:

```
O = floor(log(s / (detail_work * base_frequency)) / log(lacunarity)) + 1
O = clamp(O, 4, 9)
```

The condition is that the finest octave's wavelength is at least `detail_work`
working pixels. For `S = 1024`, `detail_px = 2`, `base_frequency = 3.5`,
`lacunarity = 2`, this gives `O = 8`.

Normalise the result. The exact range depends on the noise implementation, so do not
assume `[-1, 1]`. A robust approach:

```
med   = median(N)
scale = max(percentile(N, 99) - med, med - percentile(N, 1))
N     = clip((N - med) / scale, -1, 1)
```

This maps the larger tail to about ±1 and is stable across seeds and libraries. For
a library whose output is already normalised (e.g. OpenSimplex), a simple
`N /= max(abs(N))` is enough; the percentile version is safer across libraries.

### 4.2 Domain warping

Warp the lookup coordinates with a low-frequency vector field:

```
Wx = fbm_warp(x + 13.7, y +  7.1)
Wy = fbm_warp(x -  4.2, y + 19.3)
x_w = x + warp_amp * Wx
y_w = y + warp_amp * Wy
N_w = N(x_w, y_w)
```

`warp_amp = warp * base_radius`, with `warp` in island radii (default `0.30`).
The warp field should be low frequency (2–3 octaves, `base_frequency` around 1.5–2.5;
the `warp_frequency` parameter in the table) so it bends large sections of coast
rather than adding more roughness.

Domain warping is the difference between "a circle with noise on the rim" and "a
coast with bays and capes". It is the single most valuable addition after the base
silhouette. It is also the easiest thing to overdo: at `warp > 0.4` island radii the
coast can fold, create thin necks, or produce disconnected fragments. Keep it below
0.3 for the natural preset.

For a cheaper approximation, warp the radial angle instead:

```
theta_w = theta + warp_angle * noise1D(theta)
```

This is faster and cannot self-intersect, but it only bends the star-shaped outline;
it cannot create a deep bay that wraps around the centre. Use it only in the fast
fallback.

### 4.3 Coastal band

Split the octaves into low and high groups:

```
N_low  = normalise(sum_{i=0..2} ...)
N_high = normalise(sum_{i=3..O-1} ...)
band   = exp(-(phi / coast_band)^2)
F = phi + amp_low * N_low + amp_high * band * N_high
```

with `coast_band` around `0.10–0.15` short-side units. If `coast_amp` is in island
radii, convert it once: `amp = coast_amp * base_radius`, then split
`amp_low = 0.6 * amp` and `amp_high = 0.4 * amp`.

Why the band: high-frequency noise everywhere creates inland lakes, speckle and a
uniform "fuzzy" texture. Multiplying the high octaves by a Gaussian around the coast
keeps the interior smooth (useful for future biomes) while making the coastline
detailed. The low octaves act everywhere, which is what creates large bays and
asymmetry.

`coast_band` is the width of the detail zone. Too small: only a thin fringe is
rough, and the coast looks outlined rather than fractal. Too large: the detail
leaks inland and creates holes. Start at `0.12` and tune visually.

### 4.4 Noise type and octave count

- Prefer Simplex/OpenSimplex over Perlin if a dependency is acceptable; the
  thresholded contours have fewer axis-aligned artefacts.
- If implementing in NumPy only, use value noise with per-octave rotations and
  domain warping; the warp hides most of the blockiness.
- Do not use more than 9 octaves at 1024 px. Octaves whose wavelength is below
  `detail_work` only add aliasing and cost.
- Do not use a single octave. The hierarchy is the point.

---

## 5. Step 4 — displaced field and sea level

The displaced field is:

```
F(x, y) = phi(x, y) + amp_low * N_low + amp_high * band * N_high
```

`F` is still in short-side units. The coastline is `F = t`.

### 5.1 Solving for coverage

`coverage` is measured against the **short-side square**, not the full canvas.
Apply the edge penalty from §5.2 to `F` first, then solve `t` on the penalized
field; solving before the penalty changes the area near the margin.

```
A_ref  = s * s                    # s = min(Gw, Gh), in working pixels
target = coverage * A_ref
q      = target / (Gw * Gh)       # target as a fraction of the full grid
t      = quantile(F, 1 - q)
```

This keeps the island's physical size consistent across aspect ratios. On a 16:9
canvas, a `coverage = 0.30` island covers 30% of the largest centred square, which
is about 17% of the full canvas. If coverage were measured against the full canvas,
a 30% target on 16:9 would require a much larger island and would collide with the
margin. If a true canvas-area fraction is needed, expose a separate
`area_basis = "canvas"` option and scale the target accordingly.

The area of `{F >= t}` is monotonic in `t`, so the threshold can be solved rather
than guessed.

Fast version (no cleanup in the loop):

```
target = coverage * s * s          # short-side square, in working pixels
q = target / (Gw * Gh)             # fraction of the full grid
t = quantile(F, 1 - q)
```

This gives the target area exactly for the raw field. Cleanup (removing tiny
components, filling pinholes) changes the area by a small amount; if exact coverage
matters, iterate:

```
for iter in range(4):
    mask = cleanup(F >= t)
    actual = count(mask) / (s * s)
    if abs(actual - coverage) < 0.01: break
    t -= (coverage - actual) * step     # step > 0; or bisect
```

Area decreases as `t` increases, so the sign matters: if the mask is too small,
lower `t`. A bisection over `t` with 8–10 iterations and cleanup inside is more
robust and still cheap at 1024 px. The coverage tolerance should be ±1–2%; tighter than that
is not visually meaningful.

### 5.2 Edge safety

The base shape should already be inset by `margin`, but noise and warping can push
the coast outward. Apply a soft edge penalty to `F` **before** solving coverage (or
re-solve coverage after applying it):

```
d_edge = min(x, y, Wc - x, Hc - y)
penalty = max(0, (margin - d_edge) / margin)^2 * edge_strength
F = F - penalty
```

with `edge_strength` around `1.0` (short-side units). This bends the coast away from
the border smoothly instead of cutting a straight line through it.

Then validate:

```
bbox = bounding_box(mask)
if bbox touches the margin frame:
    reduce base_radius by 5% and retry (max 3)
    or shift the centre toward the canvas centre and retry
```

If the island still touches after the retries, reject the seed and report it. It is
better to regenerate than to emit an island with a flat edge.

### 5.3 Multiple islands

For `islands = N`:

1. Generate `N` base shapes with centres from a Poisson-disk-like distribution, or
   by rejection sampling with a minimum spacing of `island_spacing * base_radius`.
2. Assign each a radius from a seeded distribution, e.g.
   `base_radius_i = base_radius * Uniform(0.55, 1.0)`.
3. Union the SDFs: `phi = min_i(phi_i)`.
4. Use the same noise and sea-level solve as for one island.
5. After cleanup, keep the `N` largest components with area ≥ `min_island_area`.

The union of disjoint SDFs is negative between islands. The noise field is shared,
so the coasts are not statistically independent, but the water gap keeps the shapes
separated until the components get close. If the noise merges two islands, that is
usually acceptable (an isthmus); if it creates a chain of tiny fragments, increase
`island_spacing` or reduce `coast_amp`.

### 5.4 Lagoons and inland water

Subtract a hole from the base shape:

```
phi = max(phi_base, -phi_hole)
```

where `phi_hole` is the SDF of an ellipse/metaball placed inside the island. The
noise can then breach the ring, creating a channel; if a closed lagoon is wanted,
keep the hole away from the coast and either reduce `coast_amp` or move the hole
further inland.

After thresholding, holes are background components enclosed by land. Fill holes
smaller than `min_hole_area`; keep the rest if `lagoons = true`, otherwise fill all.
See §6.3.

---

## 6. Step 5 — binary mask and cleanup

```
mask = (F >= t).astype(uint8) * 255
```

### 6.1 Tiny morphology

A 3x3 elliptical open followed by a 3x3 close is usually enough:

```
mask = morphologyEx(mask, MORPH_OPEN,  ellipse(3))
mask = morphologyEx(mask, MORPH_CLOSE, ellipse(3))
```

- Opening removes single-pixel spikes and specks.
- Closing fills single-pixel pinholes and hairline cracks.
- Do **not** use a large kernel to "make it smoother". That is what the contour
  smoothing is for; a large kernel deletes capes and narrow channels.

If the mask is already clean (subpixel contour, no speckle), skip morphology
entirely. It is a repair step, not a shaping step.

### 6.2 Connected components

```
num, labels, stats, centroids = connectedComponentsWithStats(mask, connectivity=8)
```

Keep:

- the largest component if `islands = 1`;
- the `N` largest components with `area >= min_island_area * s * s` if
  `islands = N`.

Remove everything else. If no component survives, reject the seed (coverage too low
or noise too strong).

This step is also the main defence against "confetti": small offshore fragments
created by high-frequency noise. If confetti is desirable (reefs, islets), add a
separate `islet_chance` parameter rather than letting noise decide.

### 6.3 Holes

Use the contour hierarchy (`RETR_CCOMP`) or a flood fill from the border:

- Background components not connected to the border are holes.
- For each hole, compute its area.
- If `area < min_hole_area * s * s` or `lagoons = false`, fill it.
- If `lagoons = true`, keep holes above the threshold.

Holes are the one place where the existing SVG writer needs attention: it emits one
path per contour without an even-odd fill rule, so a filled island with a lagoon
would render the lagoon as land. Outline-only output is unaffected. See §9.3.

---

## 7. Step 6 — contour extraction

There are two practical choices.

### 7.1 `cv2.findContours` (simplest, already in the repo)

```
contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
```

- Returns pixel-centre chains; with `CHAIN_APPROX_NONE` every boundary pixel is a
  point.
- `RETR_CCOMP` gives a two-level hierarchy: outer contours and holes.
- Fast, no new dependency, and already used by `code/vectorization.py` / `cli.py`.
- Downside: the contour is on pixel centres, so a polygon through it is about half a
  pixel inside the ink on every side. The repo's README documents this as the
  "strict IoU ceiling". For an outline that is usually invisible; for a filled
  shape it slightly shrinks the island.

### 7.2 Marching squares (subpixel, better for SVG)

`skimage.measure.find_contours(F, level=t)` implements marching squares on the
scalar field and returns floating-point `(row, col)` contours
([scikit-image docs](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours)).
The algorithm walks each 2x2 cell, builds a 4-bit case index, linearly interpolates
the crossing along the cell edges, and stitches the segments into closed loops
([marching squares](https://en.wikipedia.org/wiki/Marching_squares)).

- Subpixel positions, so no half-pixel bias and no stair steps.
- The contour follows the actual field, not the rasterised mask, so it is smoother
  at the same resolution.
- Saddle-point ambiguity (cases 5 and 10) must be resolved consistently; using the
  cell-centre average is the usual rule.
- If `scikit-image` is not acceptable as a dependency, the 16-case table is small
  enough to implement directly; it is more code than `findContours` but not much.

**Recommendation:** use `findContours` for the first working version and march
squares for the quality version. The rest of the pipeline (simplify, smooth, SVG)
is identical.

For a mask with holes, classify contours by signed area and containment: outer
contours are counter-clockwise, holes clockwise (or the reverse, depending on the
image coordinate convention). Keep the hierarchy so the SVG can use even-odd fill.

---

## 8. Step 7 — simplify and smooth

### 8.1 Resample by arc length

Before simplifying, resample the contour to roughly uniform spacing:

```
step = max(1.0, detail_work / 2)     # working pixels
```

This prevents RDP and Chaikin from behaving differently on long straight runs and
tight clusters of pixel points.

### 8.2 RDP simplification

```
approx = cv2.approxPolyDP(contour, epsilon=simplify_px, closed=True)
```

`simplify_px` is in working pixels; start at `0.6` and do not exceed about `1.5`.
RDP removes points that are within `epsilon` of the line between their neighbours.
Too large an epsilon flattens bays and produces a polygonal coast; too small leaves
pixel noise for the smoother to deal with.

RDP can self-intersect on a closed curve if the tolerance is large relative to a
narrow neck. Validate after simplification; if the polygon is no longer simple,
reduce `epsilon`.

### 8.3 Chaikin corner cutting

For each edge `P_i -> P_{i+1}` of a closed polygon, replace it with two points:

```
Q_i = 0.75 * P_i + 0.25 * P_{i+1}
R_i = 0.25 * P_i + 0.75 * P_{i+1}
```

The new sequence is `Q_0, R_0, Q_1, R_1, ...`, still closed. One or two iterations
are usually enough. This is the corner-cutting subdivision scheme introduced by
Chaikin (1974); it converges to a quadratic B-spline and produces a smooth curve
that stays close to the original polygon.

Properties and caveats:

- It rounds convex corners and shrinks the shape slightly. For a 1-iteration cut on
  a convex polygon, the area loss is small; for a very spiky contour it can be
  noticeable. Compensate by solving the coverage slightly high (e.g. +1%) or by
  scaling the smoothed polygon around its centroid by `sqrt(A_before / A_after)`.
- It does not add detail; it only removes sharp corners. If the contour is already
  smooth, skip it.
- On a very narrow neck, repeated smoothing can close the neck or create a
  self-intersection. Validate; reduce iterations if needed.

### 8.4 Alternatives

- **Cyclic Gaussian smoothing** along the contour (treat x and y as periodic
  signals, convolve with a Gaussian of `sigma` points): less shrinkage than Chaikin,
  but can round capes more aggressively.
- **Centripetal Catmull–Rom** through the simplified points: interpolates the points
  and gives a smooth curve, but can overshoot in tight bays; the centripetal variant
  avoids the worst loops
  ([Catmull–Rom](https://en.wikipedia.org/wiki/Catmull%E2%80%93Rom_spline)).
- **Cubic Bézier fitting** with the repo's existing fitter: the best final quality,
  and it is already implemented. Feed it a lightly smoothed contour; it will choose
  the curve segments and control points. The fitter's curvature segmentation can
  treat fractal noise as many tiny runs, so pre-smoothing is important.

**Recommendation:** RDP at `0.6 px`, one Chaikin iteration, then the existing Bézier
fitter (or direct SVG with cubic curves). This removes raster stair steps while
keeping the silhouette and most of the small bays.

### 8.5 Optional re-detail

If the smoothed outline looks too soft, add a small fractal displacement along the
normals after smoothing:

```
P_i' = P_i + n_i * A * noise1D(s_i / lambda)
```

with `A` around `0.5–1.5` working pixels and `lambda` around `8–32` pixels. Taper
`A` near tight concavities to avoid self-intersections, and validate afterwards.
This is optional; the heightfield already contains detail, and Chaikin only removes
the pixel-level part.

---

## 9. Step 8 — output

### 9.1 Raster mask

Write an RGBA PNG:

- land: `(0, 0, 0, 255)` or any colour with alpha 255;
- water: alpha 0;
- optionally antialias the edge by supersampling 2x and downsampling with area
  interpolation.

The existing CLI thresholds `alpha > 252`, so the interior must be fully opaque;
antialiased edge pixels below 252 are treated as water. That is fine for an outline,
but if the antialiased edge matters, threshold at 128 in a dedicated `--mask`
mode or output a hard mask.

The shipped generator defaults to an **outline PNG** instead: the alpha channel is
the constant-width stroke band and the island interior is transparent. `--filled`
writes the solid land mask described above. The same distance-field stroke is used
for the outer coastline and for any subdivision borders.

### 9.2 SVG

Scale the working-grid points to the requested `width x height`:

```
X = x_work * (width  / Gw)
Y = y_work * (height / Gh)
```

Write `viewBox="0 0 width height"`, one `<path>` per landmass, with holes as
additional subpaths and `fill-rule="evenodd"` if filled. For outline-only, a single
closed path per component is enough.

If the existing Bézier fitter is used, the output is already in the repo's SVG
format; if not, convert the smoothed polygon to cubic segments or write a polyline
first and fit later.

### 9.3 The hole/fill-rule caveat

The existing `fileutils.saveCons` writes one filled path per contour at 75% opacity
and does not set `fill-rule="evenodd"`. For an island with a lagoon, the hole
contour would be painted as land on top of the island. Options:

1. Outline-only output (no fill): holes are just inner outlines; no problem.
2. Fill with even-odd: combine outer and hole subpaths into one path.
3. Fill with separate paths: paint the island, then paint the lagoon with the water
   colour on top. Simple, but not a real hole for export.

This is a pre-existing limitation, not something the island generator should solve
silently.

### 9.4 Metadata

For reproducibility and tests, record:

- seed and all parameters;
- working resolution;
- coverage before/after cleanup;
- component count, hole count;
- bounding box and margin;
- estimated fractal dimension;
- runtime.

This makes it possible to compare runs and to store golden seeds.

### 9.5 Image operations, in order

Most of the pipeline is image processing rather than noise generation. The order
matters as much as the operation.

| # | Operation | Typical setting | Purpose | Main risk |
|---:|---|---|---|---|
| 1 | Rasterise base silhouette | working grid, hard mask | turn the analytic shape into pixels | aliasing at the edge |
| 2 | Distance transform (inside + outside) | `DIST_L2`, mask 5 | signed distance field | half-pixel bias; smooth lightly |
| 3 | Gaussian blur on `phi` | sigma 0.5–1.0 working px | remove raster quantisation | too much blur shrinks bays |
| 4 | Domain warp / remap | `warp` 0.0–0.3 radii | bend the coast lookup | folds and thin necks at high values |
| 5 | Threshold / quantile | solved `t` | binary land/water | wrong basis gives wrong area |
| 6 | Morphological open + close | 3x3 ellipse | remove specks / pinholes | large kernels delete capes |
| 7 | Connected components | 8-connectivity | keep main island(s) | removing a legitimate islet |
| 8 | Hole fill | area threshold | remove pinholes / keep lagoons | fill-rule issue in SVG |
| 9 | Resize / supersample | 2x then area downsample | output resolution | blurring fine detail |
| 10 | Contour extraction | marching squares or `findContours` | vector boundary | pixel-centre bias |
| 11 | RDP simplify | 0.3–1.0 px | fewer points | removes small bays |
| 12 | Chaikin / cyclic Gaussian | 1–2 iterations | remove stair steps | shrinkage, over-rounding |
| 13 | Optional normal displacement | 0.5–1.5 px, tapered | restore fine detail | self-intersections |
| 14 | Bézier fit | existing fitter | smooth SVG | overshoot in tight bays |

Two rules of thumb:

- **Do not use a large blur or a large morphology kernel to fix a noisy silhouette.**
  Fix the noise amplitude or the octave count instead; large kernels destroy the
  features you are trying to keep.
- **Do not smooth before thresholding and then again after contouring.** Pick one
  stage for the main smoothing, usually the contour stage, and keep the field-level
  blur minimal.

---

## 10. Parameter table

Names are suggestions for the future API. Ranges are guidance, not hard limits.

| Parameter | Unit | Range | Default | Effect |
|---|---|---:|---:|---|
| `width`, `height` | px | > 0 | required | output canvas |
| `seed` | int | 0..2^32-1 | random | all random choices |
| `coverage` | fraction of short-side square | 0.05–0.55 | 0.30 | land area |
| `margin` | fraction of short side | 0.02–0.15 | 0.08 | water gap at the border |
| `base_radius` | fraction of short side | 0.15–0.44 | 0.30 | base island size before sea-level solve |
| `aspect` | ratio | 0.5–3.0 | 1.2 | elongation; preserves base area |
| `rotation` | degrees | 0–360 | random | orientation of the long axis |
| `center_jitter` | fraction of short side | 0–0.05 | 0.02 | how far the centre may move |
| `lobes` | count | 2–6 | 5 | number of low-frequency lobe harmonics (approximate) |
| `lobe_strength` | fraction | 0–0.35 | 0.30 | how non-circular the base is |
| `hurst` | 0–1 | 0.55–0.85 | 0.74 | fBm roughness; lower = rougher |
| `octaves` | count | 4–9 | auto | number of noise scales; the CLI caps at 7 for batch speed |
| `lacunarity` | ratio | 1.8–2.2 | 2.0 | frequency step between octaves |
| `base_frequency` | cycles per unit | 2.0–5.0 | 3.5 | size of the largest noise features |
| `warp` | island radii | 0–0.4 | 0.32 | domain-warp strength |
| `warp_frequency` | cycles per unit | 1.0–4.0 | 2.0 | scale of the warp field |
| `coast_amp` | island radii | 0.02–0.25 | 0.18 | how far the coast can move |
| `coast_band` | short-side units | 0.05–0.25 | 0.15 | width of the high-detail coastal zone |
| `detail_px` | output px | 1.0–6.0 | 2.0 | smallest feature worth generating |
| `smooth` | iterations | 0–3 | 1 | Chaikin passes |
| `simplify_px` | working px | 0.2–1.5 | 0.6 | RDP tolerance |
| `islands` | count | 1–12 | 1 | number of landmasses |
| `island_spacing` | island radii | 0.9–5.0 | 1.0 | minimum separation before shrinking |
| `subdivisions` | count | 0–12 | 0 | regions per island; 0 = no internal borders |
| `stroke_width` | px | 0 = auto | 0 | constant outline width |
| `filled` | bool | — | false | solid fill instead of the default outline |
| `lagoons` | bool | — | false | keep large enclosed water bodies |
| `min_island_area` | fraction of short-side square | 0.001–0.02 | 0.005 | discard smaller components |
| `min_hole_area` | fraction of short-side square | 0.0005–0.01 | 0.002 | fill smaller holes |
| `epsilon_area` | fraction | 0.005–0.02 | 0.01 | coverage tolerance |

Notes:

- `epsilon_area` is the coverage tolerance used by the solve/cleanup loop; `0.01`
  means ±1% of the short-side square.
- `hurst` and `coast_amp` interact. Higher `hurst` (smoother) needs less smoothing;
  lower `hurst` (rougher) needs more care with thin features and cleanup.
- `coverage` and `base_radius` interact. `base_radius` sets the rough footprint;
  the sea-level solve makes the final area match `coverage`. A useful first guess is
  `base_radius ≈ sqrt(coverage / pi)` (about `0.309` for `coverage = 0.30`), then
  adjust for lobes and aspect. If `coverage` is much larger than the base footprint
  can support, the solve pushes the coast outward into the margin; the edge safety
  then fights it. Keep `base_radius` in the range where the base area is within
  about ±40% of the target area.
- `aspect` preserves base area by using `sqrt(aspect)` on each axis. A very high
  aspect makes a long, thin island; check `min_width` before accepting.

---

## 11. Presets

These are starting points, not final values.

### Natural (default)

```
coverage 0.30, margin 0.08, base_radius 0.30, aspect 1.2, lobes 5,
lobe_strength 0.30, hurst 0.74, warp 0.32, coast_amp 0.18,
coast_band 0.15, smooth 1, simplify_px 0.6
```

### Smooth / cartographic

```
hurst 0.82, warp 0.15, coast_amp 0.10, coast_band 0.12,
smooth 2, simplify_px 0.8, lobes 4, lobe_strength 0.20
```

Less high-frequency detail, more readable at small sizes. Good for a printed-map
look.

### Rugged / fjord

```
hurst 0.62, warp 0.35, coast_amp 0.24, coast_band 0.16,
aspect 1.5, smooth 0–1, simplify_px 0.4, lobes 5, lobe_strength 0.35
```

More high-frequency energy and stronger warping; expect more thin necks and
fragments, so keep `min_island_area` and the component cleanup strict.

### Archipelago

```
islands 5, coverage 0.25 (total), island_spacing 3.0,
base_radius 0.22, aspect 1.0–1.5, hurst 0.74,
coast_amp 0.18, min_island_area 0.003
```

Generate centres with Poisson-disk-like spacing, vary radii, union the SDFs, then
keep the largest `N` components. If the result is too sparse, reduce spacing or
increase per-island radius; if it merges into one blob, increase spacing or reduce
`coast_amp`.

### Atoll / lagoon

```
base shape: ring (outer radius 0.34, inner radius 0.18, subtracted)
lagoons true, hurst 0.72, warp 0.12, coast_amp 0.10,
smooth 1, min_hole_area 0.002
```

The noise can breach the ring; if a closed lagoon is required, reduce `coast_amp` or
increase the ring width. This is the most topology-sensitive preset.

### Barrier island

```
coverage 0.10, aspect 3.5, base_radius 0.18, hurst 0.72,
warp 0.10, coast_amp 0.08, smooth 2, simplify_px 0.8
```

Check the minimum width after smoothing; a long thin island can disappear or break
into segments. `min_width` validation is strongly recommended for this preset. A
high `aspect` multiplies `base_radius` on the long axis, so keep `base_radius` small
enough that `base_radius * sqrt(aspect) + amplitude` stays inside the margin.

---

## 12. Failure modes and how to handle them

| Symptom | Likely cause | Fix |
|---|---|---|
| Island touches the canvas edge | base too large, noise too strong, centre too close | inset base, apply edge penalty, retry with smaller radius |
| Many tiny fragments | high-frequency noise, low `min_island_area` | keep largest component, raise threshold, reduce `coast_amp` |
| One blob, no bays | `lobe_strength` too low, `warp` too low | increase lobes/warp; add low-frequency asymmetry |
| "Hairy coin" | base is a circle, detail is uniform | add lobes, warp, coastal band; reduce global high-frequency noise |
| Coast looks melted / soft | `hurst` too high, smoothing too strong | lower `hurst` (raise gain), reduce `smooth`, reduce `simplify_px` |
| Coast looks like static | `hurst` too low, too many octaves, detail below pixel size | raise `hurst`, cap octaves, increase `detail_px` |
| Inland lakes everywhere | high-frequency noise not banded | use coastal band, fill small holes |
| Thin necks / self-intersections | warp/coast amplitude too high, smoothing on a narrow neck | reduce amplitudes, validate simplicity, reduce smoothing |
| Coverage far from target | sea level guessed, cleanup removing area | solve threshold, iterate with cleanup |
| Different results for same seed | unseeded RNG/permutation, hash randomisation | seed every octave/lobe/warp offset from one RNG |
| Grid-aligned artefacts | Perlin/value noise axes, octaves aligned | rotate each octave, offset, use Simplex/OpenSimplex |
| Different look at different sizes | fixed octave count, detail not scale-aware | choose octaves from `detail_px`, generate at fixed working resolution |

### Degenerate seeds

A small fraction of seeds will produce a bad island even with good parameters:

- the island is too small after cleanup;
- the island touches the border after retries;
- an atoll ring closes completely or breaks into too many pieces;
- a barrier island breaks into a chain.

The generator should validate and either retry with a derived seed (recorded in the
metadata) or return a clear failure. Silent bad output is worse than a retry.

---

## 13. Integration with this repository

### 13.1 Raster path (lowest risk)

1. Generate `mask` at `Gw x Gh`.
2. Resize to `width x height` with `INTER_NEAREST` (hard mask) or supersample and
   downsample for antialiasing.
3. Write RGBA PNG with alpha 255 on land, 0 on water.
4. Run the existing CLI:

```
python code/cli.py island.png --mask alpha --curve bezier --report
```

This reuses the contour tracer, segmentation, Bézier fitter, SVG writer and
accuracy report. The generated island is just another hand-drawn map as far as the
CLI is concerned.

### 13.2 Direct SVG path (faster, more control)

1. Extract the contour from `F` or `mask`.
2. Simplify/smooth.
3. Scale to `width x height`.
4. Write the SVG directly, or feed the points to the existing fitter.

This avoids the PNG round trip and gives control over subpixel contours and hole
handling. It duplicates some of the CLI's logic, so it should be a second step, not
the first.

### 13.3 What not to change yet

- Do not add a CLI flag or web endpoint until the generator's output is stable.
- Do not change the SVG writer's fill rule just for islands; that is a separate
  decision that affects existing maps.
- Do not add a noise dependency to `code/requirements.txt` before benchmarking it
  against the pure NumPy/OpenCV option.

---

## 14. Suggested implementation phases (for later)

**Phase 0 — prototype, outside production code.** 256/512 px, 20 seeds, compare
Perlin vs OpenSimplex vs value noise, with and without SDF. Visual review. Measure
runtime.

**Phase 1 — mask generator.** Implement the parameter table, coverage solve, edge
validation, component/hole cleanup. Tests: determinism, coverage, margin, component
count, no empty mask.

**Phase 2 — contour and SVG.** Implement `findContours` first; compare with marching
squares. Fit with the existing Bézier fitter. Score with `accuracy.py`.

**Phase 3 — presets and integration.** Add natural/smooth/rugged/archipelago/atoll
presets. Wire into CLI or web app only after the output is stable.

**Phase 4 — topology extensions.** Archipelagos, lagoons, atolls, barrier islands.
These are the cases where the SDF union/subtraction pays off.

**Phase 5 — performance and polish.** Benchmark at 512/1024/2048/4096; cache noise
permutation tables; consider spectral synthesis if octave banding is visible.

---

## 15. Complexity and runtime

Let `N = Gw * Gh`, `O` = octaves, `P` = contour perimeter in points.

| Step | Complexity | Notes |
|---|---|---|
| Base mask | O(N) | ellipse + lobes, vectorised |
| Distance transform | ~O(N) | OpenCV, fast |
| fBm | O(N * O) | dominant cost; 8 octaves at 1024² ≈ 8M samples |
| Domain warp | O(N) + 2–3 fBm evaluations | or reuse a low-res warp field |
| Threshold + quantile | O(N log N) or O(N) | quantile is fine at 1024² |
| Cleanup | O(N) | morphology + connected components |
| Contour | O(P) | findContours / marching squares |
| RDP | O(P²) worst, ~O(P log P) typical | P is small after mask contour |
| Chaikin | O(P * iterations) | trivial |
| Bézier fit | depends on fitter | usually the slowest part of the existing pipeline |

At 1024², a vectorised NumPy/OpenCv implementation should be well under a second;
the existing Bézier fitter is likely to dominate the end-to-end time. If the noise
is pure Python (e.g. `opensimplex` scalar calls), it can be much slower; use the
array API or generate at 512² for previews.

Memory: a handful of float32 arrays of `N` elements (4 MB each at 1024²), plus the
distance transform and labels. Under 100 MB comfortably.

---

## 16. Summary of the recommendation

- **Shape:** rotated ellipse + 3 low-frequency lobes; optional metaball union for
  archipelagos.
- **Representation:** signed distance field, inset by the margin.
- **Detail:** 6–7 octaves at 1024 px (the design range is 4–9) of
  Simplex/OpenSimplex-style fBm with `hurst ≈ 0.74` (`gain ≈ 0.60`), domain-warped,
  with high frequencies banded to the coast.
- **Size:** solve the sea level for the target coverage; apply a soft edge penalty;
  validate the margin.
- **Cleanup:** keep the largest component, fill tiny holes, tiny open/close only.
- **Outline:** marching squares if quality matters, `findContours` for the first
  version.
- **Finish:** RDP at ~0.6 px, one Chaikin pass, then the existing Bézier fitter.
- **Validate:** coverage, margin, components, holes, minimum width, simplicity,
  fractal dimension, determinism.

This is deliberately a composition of simple, well-understood pieces. There is no
single "island algorithm" that does all of this well; the value is in the ordering
and in the separation between macro shape, fractal detail and vector finishing.

---

## 17. Fast fallback: direct outline only

If the first version only needs an outline (no interior, no lagoons, no
archipelagos), the SDF/raster/contour stages can be skipped entirely. This is the
cheapest useful generator and a good way to validate the look before building the
full pipeline.

```
1. Sample N angles around the centre.
2. r(theta) = R * (1 + sum_k a_k * cos(k*theta + phi_k))
3. Add 1D fractal detail: r += A * fbm_1d(theta)  (periodic)
4. Convert (r, theta) to points; optionally displace along normals with
   midpoint displacement scaled by edge_length^H.
5. Chaikin once or twice; write the path.
```

- **Pros:** very fast, no raster step, guaranteed simple closed curve, exact area
  control from `R`, easy to make periodic and deterministic.
- **Cons:** star-shaped only; no deep bays that wrap around the centre; no holes or
  separate islands; the result tends to look like a "spiky potato" unless the
  low-frequency harmonics are chosen carefully.
- **When to use:** icons, previews, test data, a minimal v1, or a "quick random
  island" button where the full pipeline would be overkill.
- **When not to use:** realistic large islands, lagoons, archipelagos, or anything
  that needs an interior heightfield later.

The direct outline can also be used as the **base silhouette** for the full pipeline:
generate the radial outline, rasterise it, then apply the SDF + noise stages. That
combines the fast shape control with the natural coastline detail.
