# 01 — Problem, Requirements and "Natural"

## 1. What we are actually generating

Input: a canvas of `width × height` pixels, a seed, and a small set of shape
parameters.

Output: a random island **outline** — ideally a raster mask with an alpha channel,
and/or one or more closed SVG paths — that looks like a coastline rather than a
noise blob.

This is not a full terrain generator. There are no biomes, rivers, mountains or
elevation contours in this phase. The heightfield is a means to an end: it gives a
natural coastline. If the project later wants interior terrain, the same field can
carry it, which is a reason to prefer a heightfield-based method over a direct
outline-only method.

### Hard requirements

| Requirement | Why it matters | How to test |
|---|---|---|
| Deterministic for a seed | users can share/reproduce an island; tests need stable fixtures | same seed → identical mask hash |
| Fits the canvas with a margin | an island cut by the image border does not read as an island | bounding box within `margin` of every edge |
| One coherent landmass by default | random fragments look like debris, not an island | connected-component count = 1 |
| Controllable land area | "small island" vs "large island" should be a parameter, not luck | measured area within ±1–2% of target |
| Controllable silhouette | elongation, orientation, number of major lobes | visual + aspect/lobe metrics |
| Controllable coast roughness | smooth cartographic coast vs rugged fjord coast | fractal dimension / curvature statistics |
| Controllable detail scale | features below a pixel are wasted; features too large look blobby | finest wavelength ≥ `detail_px` |
| No self-intersections | the SVG path must be a valid simple polygon (or known holes) | segment-intersection check |
| No pixel stair steps | the vector output should look drawn, not rasterised | curvature/tangent continuity after smoothing |
| Usable by the existing pipeline | `code/cli.py` expects ink in the alpha channel | generated PNG converts cleanly to SVG |
| Fast enough to iterate | parameters are tuned visually; a 30 s wait kills that | target runtime per size |

### Soft requirements

- Optional archipelagos (2–8 islands) with natural spacing.
- Optional lagoons, inland lakes and offshore islets.
- Optional "style" presets: natural, smooth/cartographic, rugged/fjord, atoll.
- Debug output (field, mask, contour) so parameters can be tuned.
- No obvious axis alignment or repeated motifs.
- Works at 16:9 and 9:16 as well as square, without stretching the island.

---

## 2. What "natural" means here

"Natural" is subjective, but it is not arbitrary. The properties below are the ones
that separate a coastline from a noisy circle, and most of them can be measured.

### 2.1 Fractal, but not infinitely rough

Real coastlines are statistically self-similar over a range of scales. The
coastline paradox is the classic statement of this: measured length grows as the
ruler shrinks, because detail exists at many scales
([Mandelbrot, 1967](https://doi.org/10.1126/science.156.3775.636);
[Wikipedia](https://en.wikipedia.org/wiki/Coastline_paradox)).

A useful summary number is the **fractal dimension** `D` of the coastline:

- `D = 1.0` — a smooth curve (circle, ellipse).
- `D ≈ 1.1–1.4` — most familiar coasts, and a good target for a natural-looking
  island.
- `D ≈ 1.5+` — fjords, rias, very broken rock coasts.
- `D ≈ 1.7` — diffusion-limited aggregation / coral-like branching; usually too
  much for a solid island.

The exact number depends on the scale range you measure over, so treat `D` as a
tuning guide, not a certificate. See
[04-evaluation-and-integration.md](04-evaluation-and-integration.md) for how to
estimate it.

### 2.2 Hierarchy of features

A natural coast is not one noise frequency. It has:

- **Macro shape (1 per island):** a long axis, a few large lobes, perhaps a curved
  "C" or a narrow waist.
- **Mesoscale (≈3–10 per island):** bays, coves, peninsulas, headlands, isthmuses,
  offshore islets.
- **Microscale (dozens to hundreds):** small bumps, rocks, inlets, roughness of the
  line itself.

If the macro shape is a circle and the micro shape is uniform noise, the result
reads as a "hairy coin". If the macro shape is controlled and the micro shape is
modulated near the coast, it reads as an island.

### 2.3 Concavity and asymmetry

A circle is the least interesting island. Natural islands are asymmetric, and many
have at least one concavity that is large relative to the island: a bay, a sound, a
channel, a lagoon. A purely radial function `r(θ)` cannot represent a deep bay that
"overhangs" in polar coordinates, because `r` must be single-valued. That is one
reason the recommended method uses a 2D field rather than a radial outline.

### 2.4 Smoothness where it should be smooth

Natural coasts are not uniformly jagged. Headlands can be sharp; beaches and bays
are often smooth arcs. A good generator produces a mix: high curvature at capes,
low curvature across bays. Uniform per-pixel noise produces the opposite: constant
jaggedness everywhere, which looks like static.

At the vector level this means:

- no 90° corners from pixel stair steps,
- no long straight runs (unless the style is intentionally angular),
- curvature that changes gradually except at deliberate capes,
- no repeated identical wiggle patterns.

### 2.5 Scale consistency

A coastline that looks good at 4096 px but turns into a fuzzy band at 512 px is not
scale-consistent. The smallest generated feature should be at least about
`detail_px` at the output size, and the number of octaves should be chosen from the
canvas size and `detail_px`, not fixed blindly. Conversely, if the output is 512 px,
generating 4096 px of detail and then downsampling just produces expensive noise.

### 2.6 What "natural" is not

| Anti-pattern | Why it fails |
|---|---|
| Circle with radial noise | the macro shape is still a circle; reads as a coin with a fuzzy rim |
| Raw Perlin/fBm threshold | land touches the border, random lakes, random component count |
| Uniform per-pixel random displacement | no scale correlation; looks like static, not a coast |
| One noise octave | either too smooth or too wiggly; no hierarchy |
| Heavy Gaussian blur on the mask | rounds away bays and peninsulas; produces a "melted" silhouette |
| Large morphological open/close | deletes thin capes and narrow channels, which are the interesting parts |
| Over-simplification (RDP ε too large) | removes small bays; coastline becomes polygonal |
| Perfect symmetry | reads as artificial immediately |

---

## 3. Why Perlin noise alone is not enough

Perlin noise (and its relatives) is a **stationary** function: every neighbourhood
has the same statistical character. That is exactly what you want for texture, and
exactly what you do not want for an island, because an island needs a notion of
"inside" and "outside".

Concretely, thresholding a raw noise field at 50% gives:

- land that runs off the canvas edge,
- an arbitrary number of disconnected components,
- an area fraction that varies wildly with the seed,
- holes and lakes wherever the noise dips,
- no control over position, elongation or orientation.

Adding a radial falloff fixes the edge and the component count, but introduces a
new problem: every island becomes a circle with noise on the rim. The macro shape
is now controlled, but it is controlled to the least interesting shape.

The fix is not a different noise function. It is a different **composition**:

1. control the macro shape explicitly (ellipse + lobes, or metaballs),
2. represent it as a distance field,
3. add multi-scale noise to the field, not to the mask,
4. warp the noise lookup so the coast bends in a non-circular way,
5. solve the sea level for the desired area,
6. clean up and vectorise.

That is the recommended pipeline in
[03-recommended-pipeline.md](03-recommended-pipeline.md).

---

## 4. Design principles

These principles are the "why" behind the pipeline; the "how" is in the next
documents.

1. **Separate macro shape from micro detail.** Parameters that change the island's
   size, position and elongation should not also change its roughness, and vice
   versa. A distance field gives that separation.
2. **Use a field, not a polygon, for the coast.** A scalar field thresholded at a
   level automatically produces closed contours, handles merging/splitting, and
   makes holes and multiple islands a topology question rather than a special case.
3. **Put the noise where it matters.** Low-frequency noise can act everywhere;
   high-frequency detail should be strongest near the coast and fade inland. This
   preserves a clean interior and avoids random interior lakes.
4. **Solve for area; do not guess the threshold.** The area of `F >= t` is monotonic
   in `t`, so a quantile or a short bisection gives the requested coverage. This is
   more robust than tuning the threshold per seed.
5. **Smooth the contour, not the field.** Field-level blur destroys bays. A light
   contour-level corner cut removes raster stair steps while keeping the silhouette.
6. **Validate and reject.** Some seeds will produce an edge-touching island, a
   two-pixel isthmus, or a fragment. It is cheaper to detect and retry than to
   handle every degenerate case.
7. **Keep every random decision seeded and named.** Octave offsets, rotations,
   lobe phases, warp offsets and centre jitter all need to derive from the seed, or
   the output will not be reproducible.
8. **Generate at a sensible resolution.** 512–1024 px is enough for shape; scale the
   resulting path to the requested canvas. Generating at 4096 px is usually wasted
   work unless the detail actually needs it.

---

## 5. Constraints from this repository

The existing pipeline is a raster-to-SVG converter, not a map generator. It imposes
a few concrete constraints:

- **Ink is found in the alpha channel by default** (`alpha > 252`). A generated
  island mask should therefore be RGBA with alpha 255 on land and 0 on water, or
  black-on-white with `--mask luminance`.
- **Contours are traced with `cv2.findContours`** (`RETR_CCOMP`, `CHAIN_APPROX_NONE`)
  and fitted with cubic Béziers. A clean, lightly smoothed contour will fit better
  than a raw pixel chain.
- **Holes are separate contours.** The current SVG writer emits one path per contour
  without an even-odd fill rule. Outline-only output is unaffected; filled output
  with lagoons will need `fill-rule="evenodd"` or a different path strategy.
- **The accuracy tool can score the result.** `code/accuracy.py` compares a raster
  mask with an SVG and reports IoU and boundary distances. This is directly reusable
  for evaluating generated outlines against their own source mask.
- **Dependencies already present:** NumPy, SciPy, OpenCV, matplotlib. No noise
  library is present. A vectorised noise implementation or a small optional
  dependency (`opensimplex`, or a FastNoiseLite binding) is required.

---

## 6. Open product questions

These change the design and should be answered before implementation:

1. Is the island a **filled landmass** or just a **line drawing**? Filled output
   needs hole handling; line output can be a single closed path.
2. Is the island **one landmass** or an **archipelago**? The latter needs island
   spacing, per-island sizing and a merge rule.
3. Are **lagoons/inland lakes** required? They are natural but complicate the SVG
   fill rule and the "one contour" assumption.
4. Is the target **realistic** or **cartographic/stylised**? The existing output is
   clean vector line art, which suggests a slightly simplified coast rather than a
   maximally fractal one.
5. Does the island need to be **regenerated interactively**? If yes, the working
   resolution and noise implementation are constrained by latency.
6. Is a **new dependency** acceptable? A pure NumPy/OpenCV version is possible but
   more code; `opensimplex` or a FastNoiseLite binding is small and higher quality.
7. Where does this live? A test-data generator, a "new map" button, or a standalone
   tool? The integration point changes the output format and the API.
