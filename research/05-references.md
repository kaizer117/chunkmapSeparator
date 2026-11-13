# 05 — References

Annotated sources used for this research. Key links were checked on 2026-11-11. The
annotations say why each source is relevant, not just what it is.

---

## 1. Noise, fractals and coastlines

**Perlin, K. (1985). "An Image Synthesizer."** *SIGGRAPH '85*, Computer Graphics
19(3), 287–296.
[doi:10.1145/325165.325247](https://doi.org/10.1145/325165.325247)
— The original gradient-noise paper. Establishes the lattice + gradient + smooth
interpolation construction that every Perlin implementation follows.

**Perlin noise — overview.** [Wikipedia](https://en.wikipedia.org/wiki/Perlin_noise)
— Accessible description of the algorithm, its history, and the classic 2D
artefacts. Useful for the "why not raw Perlin threshold" argument.

**Simplex noise — overview.** [Wikipedia](https://en.wikipedia.org/wiki/Simplex_noise)
— Explains the simplicial grid, the reduced directional artefacts, and the patent
history. The patent expired on 8 January 2022, which is why the licensing objection
to Simplex is now historical.

**Gustavson, S. — simplex noise reference implementation.**
[PDF](https://github.com/stegu/perlin-noise/blob/master/simplexnoise.pdf)
— A compact, well-known implementation note; useful when comparing 2D/3D Simplex
cost and quality.

**OpenSimplex noise (Python package).**
[PyPI](https://pypi.org/project/opensimplex/) — Patent-free gradient noise with a
NumPy array API (`noise2array`). A practical option if a small dependency is
acceptable; the page also explains the patent motivation.

**FastNoiseLite.** [GitHub](https://github.com/Auburn/FastNoiseLite) — Portable,
MIT-licensed noise library with Perlin, Simplex, value, cellular, fractal and
domain-warp variants. A good reference for parameter names and for a C/C++/C#/JS
implementation.

**Mandelbrot, B. B. (1967). "How Long Is the Coast of Britain? Statistical
Self-Similarity and Fractional Dimension."** *Science* 156(3775), 636–638.
[doi:10.1126/science.156.3775.636](https://doi.org/10.1126/science.156.3775.636)
— The founding paper on coastline fractal dimension; the reason `D` is a meaningful
target for a natural coast.

**Coastline paradox — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Coastline_paradox) — Clear explanation of
why coastline length depends on the ruler, with the Richardson/Mandelbrot history.
Useful background for the "fractal, but not infinitely rough" design goal.

**Fractional Brownian motion — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Fractional_Brownian_motion) — Defines the
Hurst exponent `H`, the covariance, and the graph dimension `2 - H`. The basis for
the gain/Hurst table in
[02-algorithm-survey.md](02-algorithm-survey.md#2-the-hurst-parameter-and-why-gain-05-is-not-enough).

**Mandelbrot, B. B., & Van Ness, J. W. (1968). "Fractional Brownian Motions,
Fractional Noises and Applications."** *SIAM Review* 10(4), 422–437.
[doi:10.1137/1010093](https://doi.org/10.1137/1010093)
— The original fBm formulation; the mathematical parent of every octave-summed
terrain function.

**Fournier, A., Fussell, D., & Carpenter, L. (1982). "Computer Rendering of
Stochastic Models."** *Communications of the ACM* 25(6), 371–384.
[doi:10.1145/358523.358553](https://doi.org/10.1145/358523.358553)
— Introduces the midpoint-displacement/stochastic-subdivision family and the
fractal-terrain approach. Relevant to the diamond–square section and the fast
fallback.

**Diamond–square algorithm — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Diamond-square_algorithm) — Describes the
algorithm, its grid creases, and the Miller/Lewis fixes. A concise summary of why it
is not the first choice here.

**Value noise — overview.** [Wikipedia](https://en.wikipedia.org/wiki/Value_noise)
— The cheapest coherent noise; useful as the "no dependency" MVP option and as a
baseline for comparison.

**Worley (cellular) noise — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Worley_noise) — Distance-to-feature-point
noise; relevant to islets, rocks and cellular detail, not the primary coast.

---

## 2. Islands and terrain generation

**Patel, A. J. — "Making maps with noise functions."** Red Blob Games, 2015.
[https://www.redblobgames.com/maps/terrain-from-noise/](https://www.redblobgames.com/maps/terrain-from-noise/)
— The best practical introduction to fBm, octaves, redistribution, water level and
the two-ingredient island recipe (distance + shaping). The island section is the
direct ancestor of the falloff discussion in this folder.

**Patel, A. J. — "Elevation Control."** Red Blob Games, 2017.
[https://www.redblobgames.com/x/1728-elevation-control/](https://www.redblobgames.com/x/1728-elevation-control/)
— Distance fields, harmonic-mean blending and designer-controlled coastlines. The
"distance field as a control surface" idea is the conceptual parent of the
recommended SDF step.

**Patel, A. J. — "Polygonal Map Generation."**
[http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/](http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/)
— The classic Voronoi/Delaunay map generator: Poisson disk points, elevation,
rivers, biomes. The reference point for the "full map generator" direction.

**Patel, A. J. — "Procedural map generation on a sphere."** Red Blob Games, 2018.
[https://www.redblobgames.com/x/1843-planet-generation/](https://www.redblobgames.com/x/1843-planet-generation/)
— Voronoi plates, random flood fill and distance-field interpolation. Useful if the
project later needs continents rather than islands.

**Guerra, C. — "Procedural Island Generation (I)."** 2025.
[https://brashandplucky.com/2025/09/07/procedural-island-generation-i.html](https://brashandplucky.com/2025/09/07/procedural-island-generation-i.html)
— Poisson disk sampling, Delaunay/Voronoi meshes, blue-noise point distribution and
the red/blue quad mesh. A detailed modern walkthrough of the polygon approach.

**Guerra, C. — "Procedural Island Generation (II)."** 2025.
[https://brashandplucky.com/2025/09/10/procedural-island-generation-ii.html](https://brashandplucky.com/2025/09/10/procedural-island-generation-ii.html)
— The "paint map" idea: a coarse grid that defines the island's base elevation,
built from a distance field plus fBm, then mountain ridges. Directly relevant to the
base-shape + field composition.

**Guerra, C. — "Procedural Island Generation (III)."** 2025.
[https://brashandplucky.com/2025/09/17/procedural-island-generation-iii.html](https://brashandplucky.com/2025/09/17/procedural-island-generation-iii.html)
— Multi-scale noise layers, coastal noise enhancement `(1 - e^4)`, distance-based
mountains and blending. The coastal-band idea in
[03-recommended-pipeline.md](03-recommended-pipeline.md) is a simplification of
this.

**"Build a Procedural Island From Scratch in One Hour."** 3D Simulations, 2026.
[https://www.mysimulator.uk/content/tutorials/procedural-island-tutorial.html](https://www.mysimulator.uk/content/tutorials/procedural-island-tutorial.html)
— A concise fBm + radial falloff + erosion tutorial. Good for the baseline recipe
and for the falloff formula `d^a / (d^a + (b - b*d)^a)`.

**Turner, S. — "Perlin Noise, Procedural Content Generation, and Interesting
Space."** Here Dragons Abound, 2019.
[https://heredragonsabound.blogspot.com/2019/02/perlin-noise-procedural-content.html](https://heredragonsabound.blogspot.com/2019/02/perlin-noise-procedural-content.html)
— A useful critique: Perlin noise is a happy accident for landscapes and does not
generalise to interesting content. Supports the "noise is a detail engine, not an
island engine" argument.

**Mapgen4.** [https://www.redblobgames.com/maps/mapgen4/](https://www.redblobgames.com/maps/mapgen4/)
— Interactive map generator with paint-map control. Useful as a reference for the
kind of parameter control a user can handle.

---

## 3. Shape alternatives

**Blinn, J. F. (1982). "A Generalization of Algebraic Surface Drawing."** *ACM
Transactions on Graphics* 1(3), 235–256.
[doi:10.1145/357306.357310](https://doi.org/10.1145/357306.357310)
— The metaball paper. Relevant to the optional base-shape generator for lobes,
archipelagos and atolls.

**Metaballs — overview.** [Wikipedia](https://en.wikipedia.org/wiki/Metaballs)
— Accessible definition, falloff functions, and the merge behaviour. Good enough for
implementing a 2D Gaussian/metaball base.

**Voronoi diagram — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Voronoi_diagram) — The dual of Delaunay;
the geometric basis of the polygon-map approach.

**Delaunay triangulation — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Delaunay_triangulation) — Maximises the
minimum angle; the standard mesh for irregular terrain.

**Bridson, R. (2007). "Fast Poisson Disk Sampling in Arbitrary Dimensions."**
*SIGGRAPH sketches*.
[PDF](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf)
— The standard O(N) Poisson-disk algorithm; the point distribution behind Voronoi
maps and archipelago island centres.

**Lloyd's algorithm — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) — Voronoi
relaxation; useful for evenly spaced island centres, though it can look too regular.

**Witten, T. A., & Sander, L. M. (1981). "Diffusion-Limited Aggregation, a Kinetic
Critical Phenomenon."** *Physical Review Letters* 47(19), 1400–1403.
[doi:10.1103/PhysRevLett.47.1400](https://doi.org/10.1103/PhysRevLett.47.1400)
— The DLA paper. The 2D cluster dimension is about 1.71, which is why it is too
branched for a solid island.

**Diffusion-limited aggregation — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Diffusion-limited_aggregation) — Clear
description of the random-walk growth and the resulting Brownian tree. Useful for
the "too fractal" verdict.

**Turing, A. M. (1952). "The Chemical Basis of Morphogenesis."** *Philosophical
Transactions of the Royal Society B* 237(641), 37–72.
[doi:10.1098/rstb.1952.0012](https://doi.org/10.1098/rstb.1952.0012)
— The origin of reaction–diffusion patterns; the theoretical basis for the
Turing-pattern alternative.

**Reaction–diffusion system — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system) —
Covers the Gray–Scott model and the pattern types. Useful for the "stylised variant"
discussion.

**Gray–Scott model.** MIT.
[https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)
— The standard two-chemical reaction–diffusion demo; a reference if the project ever
wants a coral/reef style.

**Runions, A., Lane, B., & Prusinkiewicz, P. (2007). "Modeling Trees with a Space
Colonization Algorithm."** *Eurographics Workshop on Natural Phenomena*, 63–70.
[https://www.algorithmicbotany.org/papers/colonization.egwnp2007.html](https://www.algorithmicbotany.org/papers/colonization.egwnp2007.html)
— The space-colonisation algorithm; relevant to rivers and dendritic peninsulas, not
to a closed island outline.

**Cellular automaton — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Cellular_automaton) — Background for the
CA-based cave/lagoon style; not recommended as the primary generator.

---

## 4. Contours, smoothing and image operations

**Marching squares — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Marching_squares) — The 16-case contouring
algorithm, linear interpolation and saddle-point disambiguation. The basis for
subpixel coastline extraction.

**scikit-image `find_contours`.**
[https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours)
— A ready-made marching-squares implementation returning subpixel contours. The
lowest-effort way to get a quality contour if adding `scikit-image` is acceptable.

**OpenCV `findContours` tutorial.**
[https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
— The function already used in this repository; explains `RETR_CCOMP` and
`CHAIN_APPROX_NONE`.

**OpenCV contour features (`approxPolyDP`).**
[https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
— RDP simplification and contour properties; the reference for the `simplify_px`
step.

**Ramer–Douglas–Peucker algorithm — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
— The algorithm, its `epsilon`, and its worst-case complexity. The warning about
self-intersection at large tolerances is relevant to narrow necks.

**Chaikin, G. M. (1974). "An algorithm for high-speed curve generation."**
*Computer Graphics and Image Processing* 3(4), 346–349.
[doi:10.1016/0146-664X(74)90028-8](https://doi.org/10.1016/0146-664X(74)90028-8)
— The corner-cutting subdivision scheme used for the light contour smoothing.

**Catmull–Rom spline — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Catmull%E2%80%93Rom_spline) — The
interpolating spline and its centripetal variant; an alternative to Chaikin and a
useful way to convert points to cubic Béziers.

**Distance transform — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Distance_transform) — The image operation
behind the SDF step; explains the signed-distance idea and common algorithms.

**OpenCV distance transform tutorial.**
[https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html](https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html)
— Practical `cv2.distanceTransform` usage and the `DIST_L2` mask sizes.

**OpenCV morphological transformations.**
[https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
— Erosion, dilation, opening, closing, and structuring elements. The reference for
the small cleanup step and for why large kernels are dangerous.

**SciPy `ndimage`.**
[https://docs.scipy.org/doc/scipy/reference/ndimage.html](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
— Gaussian filtering, labelling, hole filling and morphology; the utilities used by
the cleanup and evaluation steps.

**Gaussian blur — overview.**
[Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur) — Background for the light
field/contour smoothing and the warning that blur radius and feature size interact.

---

## 5. Tools and libraries

**NumPy.** [https://numpy.org/](https://numpy.org/) — Array operations; the
foundation of a vectorised field implementation.

**OpenCV.** [https://opencv.org/](https://opencv.org/) — Distance transform,
morphology, connected components, contours and resizing. Already a project
dependency.

**scikit-image.** [https://scikit-image.org/](https://scikit-image.org/) — Marching
squares, labelling and morphology; optional, but the shortest path to subpixel
contours.

**FastNoiseLite.** [GitHub](https://github.com/Auburn/FastNoiseLite) — Portable
noise library; useful as a quality/speed reference even if the project stays in
Python.

**OpenSimplex (Python).** [PyPI](https://pypi.org/project/opensimplex/) — A small
NumPy-friendly noise dependency if the pure NumPy implementation is not good enough.

**`noise` (Python).** [PyPI](https://pypi.org/project/noise/) — The classic Perlin
package; old but simple. Mentioned for completeness; check maintenance and Python
version support before using.

---

## 6. Further reading

- **The Book of Shaders — Fractal Brownian Motion.**
  [https://thebookofshaders.com/13/](https://thebookofshaders.com/13/) — Visual
  explanation of fBm, octaves and gain; good for tuning intuition.
- **Inigo Quilez — "Warp."**
  [https://iquilezles.org/articles/warp/](https://iquilezles.org/articles/warp/) —
  The canonical domain-warping article: `f(p + fbm(p))`. The recommended pipeline's
  warp step is a 2D adaptation of this idea.
- **Inigo Quilez — articles index.**
  [https://iquilezles.org/articles/](https://iquilezles.org/articles/) — fBm,
  warping, noise, shaping functions and distance fields; a deep reference for the
  procedural side.
- **Red Blob Games — noise tutorial.**
  [https://www.redblobgames.com/articles/noise/introduction.html](https://www.redblobgames.com/articles/noise/introduction.html)
  — Frequency, wavelength, amplitude, octaves and noise colours; useful background
  for the parameter names.
- **Runevision — erosion filters.**
  [https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html)
  — If the island later needs eroded valleys and beaches, this is a practical
  starting point; out of scope for the outline phase.
