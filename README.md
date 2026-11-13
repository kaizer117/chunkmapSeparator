<h1>Chunk Map Separator</h1>

<p style="text-align:center">
<cite>The good cartographer is both a scientist and an artist. </cite>
<br><b>Erwin Josephus Raisz (1893 Ã¢â‚¬â€œ 1968)</b>
</p>


<img src="resources/read_me_rcs/readme_banner.png" style="display:block;margin:auto;width: 90%; height: 90%"> <br>
<p>
Chunk Map Seprator is a convenient piece of software that can help with creating high-quality maps with beautiful procedurely generated color palattes. Primarily written in python, this software allows you to beautify your maps, create visually striking details and customize color palattes to your hearts content.<br><b>Note: This project is still WIP with many fetaures yet to come</b> </p>

## What it does

The core job is one thing: **take a hand drawn line, in a raster image, and turn
it into a clean vector outline.**

1. The ink is found from the PNG alpha channel (or from luminance, if the image
   has no usable transparency).
2. The ink boundary is traced into contours.
3. Each contour is cut into runs and a cubic bezier is fitted to every run.
4. The curves are written out as a single closed SVG path per contour.
5. The result is measured against the input so you can see how good the
   conversion actually was.

Everything above is available from one command line tool, `main.py`;
`main.py orchestrate` wraps the whole loop in a batch integration test.

For maps where the region topology is known, there is also a **region-aware**
path (`--regions LABELS`). It takes a single-channel integer label map and writes
one filled path per region. Adjacent regions share the exact same border curves,
so the thickness of the drawn line is not reproduced as a gap or a doubled line.
This is the mode used by the island integration test; see
[Integrated vectorisation test](#integrated-vectorisation-test).

## Repository layout

The repository is split so each kind of file has one obvious home:

```
chunkMapSeparator/
  main.py                 entry point: python main.py ... (the only thing you run)
  chunkmap/               operational code
    cli.py                argument parser + dispatch for all four modes
    accuracy.py           coverage / boundary measurement and the standalone scorer
    vectorization.py      cubic bezier fitting, control-point formatting
    segmentation.py       curvature-aware and blind contour splitting
    region_vectorization.py  label map -> shared-border region fills
    island_generator.py   deterministic random islands + label maps
    orchestration.py      batch runner, config loader, overlay and reports
    fileutils.py          SVG document writer (SVGHandler)
    colorspace.py         HSL palette generation
    cvutils.py            alpha histogram and contour reshaping
    programutils.py       numeric helpers
    experimental/         research fitters and scratch scripts (not on the CLI path)
  tests/
    unit/                 one code file at a time (test_accuracy.py, test_vectorization.py, ...)
    integration/          the real CLI and batch runner end to end
    conftest.py           shared scratch dir + matplotlib cache setup
    helpers.py            shared image fixtures
  configs/                sample YAML and JSON orchestration configs
  scripts/                echo-only help pages about the README commands
  help.bat                menu that opens those help pages
  resources/              example images and reference SVGs
  research/               design notes and diagrams
  web-app/                static browser shell + the placeholder Flask mock
  outputs/                generated SVGs, reports, overlays (gitignored)
```

The rule of thumb: **operational code lives under `chunkmap/`**, **unit tests
live next to the code they test under `tests/unit/`**, and **integration tests
that drive `python main.py` live under `tests/integration/`**. If you add a new
stage to the pipeline, it belongs in `chunkmap/` with a matching
`tests/unit/test_<module>.py`.

For a guided tour of the commands without reading this whole file, run:

```bat
help.bat
```

or open one of the pages in `scripts/` directly.

## Setup

The repository already contains a working `venv/` with every dependency
installed (Python 3.13). To use it:

```bat
setup.bat
```

which activates the virtual environment and leaves you at the repository root. Manually:

```bat
venv\Scripts\activate
```

Only if you are setting up a fresh clone without the bundled environment:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You no longer need to create an `outputs/` folder by hand; the tool creates it
the first time it writes a file.

## Command line tool

Run it from the repository root. Only `png` input is expected, and the ink is
taken from the alpha channel by default.

```bat
python main.py resources\example-cuba.png
```

```
chunkmap-separator  raster -> bezier svg
input      resources\example-cuba.png  (4096x3072, 4 channels)
alpha      present  min 0  max 255  mean 1.7  transparent 99.2%  opaque 0.5%  -> usable
mask       alpha  ->  66589 px (0.529% of image)
contours   66 kept, 0 skipped, 0 failed
vector     66 paths / 3157 curves / 9537 control points
timing     fit 56.07s  svg 0.02s  report 1.67s  total 58.05s
svg        156.9 KB  ->  outputs\cuba-bezier.svg
  coverage   IoU 0.9655   Dice 0.9824   precision 0.9991   recall 0.9663   (1 px tolerance)
             strict IoU 0.6651   ceiling 0.6995 from the exact raster contour   -> 95.1%
  boundary   raster->vector  mean 0.07 px   p95 0.95 px   max 2.87 px
             vector->raster  mean 0.17 px   p95 0.95 px   max 4.11 px
             hausdorff 4.11 px   within 1 px 99.9%   within 2 px 100.0%
  compress   30333 raster points -> 3157 segments (9.6x)   ink 66589 px -> 46128 px filled
```

### Usage

```
python main.py [-h] [-o PATH] [--curve {bezier,polygon}]
                   [--segmentation {curvature,blind}] [--min-chunk MIN_CHUNK]
                   [--max-chunk MAX_CHUNK] [--smooth-sigma SMOOTH_SIGMA]
                   [--chunk-size CHUNK_SIZE]
                   [--min-contour-points MIN_CONTOUR_POINTS] [--min-area MIN_AREA]
                   [--mask {auto,alpha,luminance}]
                   [--alpha-threshold ALPHA_THRESHOLD]
                   [--luminance-threshold LUMINANCE_THRESHOLD] [--invert]
                   [--stroke STROKE] [--stroke-width STROKE_WIDTH]
                   [--fill {none,palette}] [--hue HUE] [--sampling {linear,random}]
                   [--picking {linear,shuffle}] [--seed SEED]
                   [--precision PRECISION] [--report | --no-report]
                   [--report-json PATH] [--quiet] [--verbose]
                   [--regions PATH] [--region-style {fill,boundary}]
                   [--region-fill {palette,solid}] [--region-palette {spread,hue}]
                   [--region-color COLOUR] [--region-seam PX]
                   [--corner-angle DEGREES]
                   input
```

The ink-band flags (`--curve`, `--segmentation`, `--mask`, `--fill`) describe the
normal contour tracer. `--regions` selects the region-aware mode instead and
ignores them.

```bat
:: region-aware SVG from a label map: 0 = water, positive = region
python main.py island.png --regions island-labels.png -o island-regions.svg

:: one single-line boundary network instead of filled regions
python main.py island.png --regions island-labels.png --region-style boundary
```

### Generate random islands

The `generate` subcommand writes a batch of random island PNGs. The default is a
**single-colour outline with a consistent stroke width**, on a transparent
background. Pass `--filled` if you want the solid-green landmass instead.

```bat
:: one outline island
python main.py generate outputs\island.png 1 1024 1024 --seed 42

:: four islands in separate areas, each divided into three regions
python main.py generate outputs\islands 8 1024 768 --islands 4 --subdivisions 3 --seed 42

:: same, but solid fill instead of an outline
python main.py generate outputs\filled 4 1024 768 --islands 3 --filled --seed 7
```

```
islands    8 x 1024x768  ->  outputs\islands
seed       42  (natural, outline, 4 per image, 3 subdivisions)
[1/8] island_0001.png  seed 42  coverage 24.4%  islands 4  regions 12  stroke 2.6  bbox 780x701
...
done       8/8 in 13.2s
```

`OUTPUT` can be a directory or a `.png` path:

```bat
:: directory: outputs\islands\island_0001.png, island_0002.png, ...
python main.py generate outputs\islands 20 2048 2048 --seed 42

:: single file, count must be 1
python main.py generate outputs\my_island.png 1 1024 1024 --style rugged

:: prefix: outputs\island_0001.png, island_0002.png, ...
python main.py generate outputs\island.png 5 512 512
```

| Option | Default | Meaning |
|---|---|---|
| `output` | required | output directory, `.png` file (count 1), or `.png` prefix |
| `count` | required | number of PNG files to write |
| `width`, `height` | required | output size in pixels |
| `--seed N` | random | base seed; file `i` uses `N + i - 1` |
| `--coverage F` | `0.30` | target land fraction of the short-side square |
| `--margin F` | `0.08` | water gap kept from the canvas edge |
| `--islands N` | `1` | islands per image, placed in separate areas (1Ã¢â‚¬â€œ12) |
| `--subdivisions N` | `0` | regions per island, 0 for none (0Ã¢â‚¬â€œ12) |
| `--stroke-width PX` | auto | outline width in pixels; auto is ~2 px at 512, ~2.5 px at 1024 |
| `--filled` | off | solid land fill instead of the outline |
| `--style {natural,smooth,rugged}` | `natural` | coastline roughness preset |
| `--quiet` | off | print only errors |

How the options behave:

- The default outline is a constant-width band centred on the coastline. It is
  drawn from a distance field, not by thickening a raster contour, so the width is
  stable along the whole coast.
- `--islands` places each island in its own area, with seeded size variation so the
  islands are not identical. When the requested coverage is too high for the
  available space, the generator may reduce the achieved coverage to keep the
  islands separate; the value printed in the summary is the real one.
- `--subdivisions` divides each island with relaxed area-aware Voronoi regions.
  Seeds are spread with farthest-point sampling, relaxed toward equal-area
  centroids, and cells smaller than about 2% of the island (or below a stroke-size
  floor) are merged into a neighbour. Very large cells are naturally pulled back by
  the centroid relaxation. The boundaries are slightly warped so they do not look
  like a ruler-drawn pie.
- Holes are filled in the first version. Lagoons and loosely clustered
  archipelagos are deferred; `--islands` places distinct, separated landmasses.
- The default outline is meant to be used directly as a drawing. Region-aware
  vectorization consumes the generator's label map, so it does not care about the
  stroke band. For the legacy ink-band converter, use `--filled` for a solid
  landmass or lower `--alpha-threshold` to about 128 for the antialiased outline.
  The converter default of 252 is tuned for hard-edged scans and shatters this
  soft outline into many small contours.

The generator uses the research pipeline in `research/`: a controlled silhouette,
a signed distance field, warped multi-octave Perlin fBm, a coverage-solved sea
level, and a small cleanup pass. It has no dependency beyond NumPy, OpenCV and
SciPy. The same seed always produces the same island.

### Integrated vectorisation test

`orchestrate` is the batch integration test. It reads a JSON or YAML config,
generates the requested random islands, vectorizes every image through the normal
CLI code path, keeps the SVGs in their own folder, renders the SVG geometry over
each island PNG in a contrasting colour, and prints the accuracy report per image
plus a batch summary.

The default test uses the **region-aware** path. The generator exposes its label
map (one positive integer per region, 0 for water); orchestration writes that map
next to the PNG and the CLI uses it instead of the antialiased stroke band. Every
boundary between two regions is fitted **once**, then reused in opposite
directions by the two region paths. Adjacent regions therefore touch exactly:
there is no second copy of the border and no gap equal to the drawn line width.

```bat
python main.py orchestrate --config configs\island-svg-sample.yaml

:: a quick two-image sample
python main.py orchestrate --config configs\island-svg-sample.yaml ^
    --count 2 --space outputs\orchestration\quick
```

The test space is given by `space` in the config or `--space` on the command
line. Relative paths are resolved from the current directory (run the commands
from the repository root, as everywhere else in this README). It is organised
like this:

```
outputs\orchestration\island-svg-sample\
  islands\island_0001.png      generated raster
  regions\island_0001.png      16-bit label map: 0 water, 1..n regions
  svg\island_0001.svg          one filled path per region, shared borders
  overlay\island_0001.png      island + SVG paths in a distinct colour
  reports\island_0001.json     full accuracy.py metrics for that image
  reports\summary.json         config snapshot, per-image digest, aggregate
  reports\summary.csv          one row per image, ready for pandas
```

The sample config, with the important fields (everything else defaults):

```yaml
name: island-svg-sample
seed: 2026
space: outputs/orchestration/island-svg-sample

generation:
  count: 8
  width: 1024
  height: 768
  style: natural            # natural | smooth | rugged
  filled: false             # false = outline line art
  coverage: 0.30
  margin: 0.08
  stroke_width: 0.0         # 0 = automatic
  islands: [1, 4]           # inclusive range, or an int
  subdivisions: [0, 4]      # inclusive range, or an int
  seed_step: 97

vectorization:
  max_chunk: 10             # maximum boundary points per fitted cubic
  stroke: "#111111"         # legacy ink-band style; unused by region fills
  stroke_width: 1.25
  precision: 2
  region_mode: true         # use the generator label map / shared borders
  region_fill: palette      # palette | solid
  region_palette: spread    # spread | hue
  region_color: "#9ec79b"   # used when region_fill is solid
  region_seam: 0.5          # same-colour hairline hiding renderer AA seams
  corner_angle: 55.0        # windowed turn (degrees) that splits a border chain

overlay:
  enabled: true
  color: "#ff3355"
  width: 1.0
  opacity: 1.0
  fill: false               # true tints the vectorised region
  fill_opacity: 0.28
  background: null

report:
  tolerance: 1
  min_iou: null             # set a number to gate the run
  min_within_1px: null
  min_relative_iou: null
  max_hausdorff_px: null
  show: detailed            # detailed | compact | summary
  json: true
  csv: true
```

`islands` and `subdivisions` can be a range (`[1, 4]`, `{min: 1, max: 4}`) or a
fixed number. The tool shuffles each range with a seeded RNG and cycles through
it, so a small sample covers every value in the range instead of accidentally
drawing the same count every time; each image still gets its own generation
seed. Use `generation.images` to pin exact cases:

```yaml
generation:
  images:
    - name: single
      seed: 101
      islands: 1
      subdivisions: 0
    - name: busy
      seed: 202
      islands: 4
      subdivisions: 4
      filled: true
```

`region_fill: palette` with `region_palette: spread` gives each region a distinct
hue. `region_palette: hue` uses the legacy `--hue`/`--sampling`/`--picking`
palette instead. `region_fill: solid` uses `region_color` for every region.

Other flags: `--space DIR`, `--seed N`, `--count N`, `--only 1,3-5`,
`--compact`, `--quiet`, `--no-overlay`, `--clean`, `--fail-fast`.

#### Why region mode exists

A normal outline PNG draws every coast and every subdivision border as a stroke
of finite width. The legacy ink-band tracer follows the alpha edge, so each
stroke becomes two parallel contours. For a coastline that is fine. For a border
shared by two regions it is wrong: the two regions get separate, offset borders
and the stroke width is visible in the SVG.

Region mode starts from the partition instead:

1. read the label map and take every lattice edge where two labels differ;
2. walk each region face (the region is kept on the left of every half-edge);
3. split the undirected boundary graph into chains at junctions;
4. fit each chain once and clamp its endpoints to the lattice junctions;
5. assemble each region path from those chains, reversing the shared ones;
6. write one filled path per region with `fill-rule="evenodd"` and no visible
   outline. A same-colour hairline (`region_seam`) covers renderer antialiasing
   seams without adding a visible border.

The result is a set of paths that touch exactly, because they literally contain
the same cubic segments. An island with no subdivisions is one region and
therefore one path; the shared-edge logic only kicks in where two labels meet.
`--region-style boundary` emits the same geography as a single stroked boundary
network instead of filled regions.

The CLI can do this outside orchestration too. A label map is a single-channel
PNG/TIFF/NPY with 0 for water and positive integers for regions:

```bat
python main.py island.png --regions island-labels.png -o island-regions.svg
python main.py island.png --regions island-labels.png ^
    --region-style boundary -o island-borders.svg
```

Region mode needs the topology. It does **not** guess regions from a stroke band:
a line could be a coastline, a shared border, or a decorative line, and the
stroke raster alone cannot disambiguate that reliably. The island generator
supplies the label map, which is why the generated workflow is exact.

#### Legacy ink-band mode in orchestration

Set `region_mode: false` to use the old stroke-band tracer with the generated
PNG. In that mode `alpha_threshold: 128` is deliberate: the generated outline is
antialiased, so the converter default 252 fragments the thin edge into hundreds
of tiny contours. 128 is the 50% coverage line. For a hard-edged scan use 252.
With `filled: true` the ink-band tracer sees an opaque landmass and follows only
the outer coast; region mode is the one that preserves subdivisions even when
the island is filled.

YAML is read with PyYAML when it is installed and with the built-in subset
parser otherwise (nested mappings and sequences, inline lists/maps, quoted
scalars, comments; no anchors, tags or block scalars). JSON always works.

A representative sample run:

```
[1/8] island_0001  seed 2026  1024x768  outline/natural
      islands  1/1  regions 3  coverage 30.0%  stroke 2.0 px
      vector   3 paths / 543 curves  svg 22.5 KB  gen 2.30s  fit 4.98s  score 0.22s
      files    islands/island_0001.png  regions/island_0001.png  svg/island_0001.svg  overlay/island_0001.png
      coverage   IoU 0.9993   Dice 0.9996   precision 0.9993   recall 1.0000   (1 px tolerance)
                 strict IoU 0.9889   ceiling 0.9880 from the exact raster contour   -> 100.0%
      boundary   raster->vector  mean 0.44 px   p95 0.95 px   max 1.91 px
                 vector->raster  mean 0.41 px   p95 0.95 px   max 4.78 px
                 hausdorff 4.78 px   within 1 px 98.4%   within 2 px 100.0%
      compress   4279 raster points -> 543 segments (7.9x)   ink 176919 px -> 177755 px filled
      status   OK

summary    8/8 images  8 ok  0 below  0 failed  wall 63.49s
coverage   IoU  mean 0.9991  median 0.9992  min 0.9987  max 0.9994  (1 px tolerance)
           Dice mean 0.9996   precision mean 0.9992   recall mean 0.9999
           strict/ceiling mean 100.0%  min 100.0%  max 100.0%
boundary   r->v mean 0.43 px  p95 mean 0.95 px  worst image mean 0.45 px
           hausdorff mean 4.80 px  worst 8.47 px  within 1 px mean 98.1%  min 97.5%
vector     4,289 curves total (536/image)  43,402 raster points  10.1x compression  183.3 KB SVG
images     worst island_0004 (IoU 0.9987)   best island_0005 (IoU 0.9994)
timing     generate 21.98s  vector 39.37s  score 1.28s  overlay 0.78s
reports    ...\reports\summary.json
           ...\reports\summary.csv
```

The per-image block comes from `accuracy.format_report`, so it reads exactly like
the standalone `accuracy.py` output. The overlay is a diagnostic drawing, not a
browser render: every parsed path is drawn as a closed loop in `overlay.color` /
`overlay.width`, with the even-odd fill optionally tinted. Exit codes: `0` all
images processed and above thresholds, `1` a conversion failure or a
below-threshold image, `2` a bad config or argument, `130` interrupted.

### Paths and basic options

| Option | Default | Meaning |
|---|---|---|
| `input` | required | source raster image (png) |
| `-o`, `--output PATH` | `outputs/<input name>.svg` | where the SVG is written |
| `--quiet` | off | print nothing except errors |
| `--verbose` | off | one line per contour while fitting |

### Where the ink is

| Option | Default | Meaning |
|---|---|---|
| `--mask {auto,alpha,luminance}` | `auto` | how to decide which pixels are ink |
| `--alpha-threshold N` | `252` | alpha **above** this counts as ink |
| `--luminance-threshold N` | `128` | luminance **below** this counts as ink |
| `--invert` | off | luminance mode: treat light pixels as ink |

`auto` uses the alpha channel when it actually varies, and otherwise falls back
to luminance (compositing transparency onto white first). This is what makes
black-on-white scans work: their alpha is 255 everywhere, which would otherwise
trace a single rectangle around the whole image.

### Vectorisation

| Option | Default | Meaning |
|---|---|---|
| `--curve {bezier,polygon}` | `bezier` | cubic curves, or the straight-line baseline |
| `--segmentation {curvature,blind}` | `curvature` | how each contour is cut into runs |
| `--min-chunk N` | `5` | curvature mode: shortest run, in raster points |
| `--max-chunk N` | `15` | curvature mode: longest run, in raster points |
| `--smooth-sigma F` | `2.0` | curvature mode: gaussian sigma for curvature |
| `--chunk-size N` | `10` | blind mode: target points per run |
| `--min-contour-points N` | `4` | drop contours shorter than this |
| `--min-area F` | `0` | drop contours with area below this, in pxÃ‚Â² |

`curvature` splits at high-curvature points, which follows hand drawn shapes
well. `blind` splits every `--chunk-size` points and is much faster. `max_chunk`
is a hard ceiling in both cases: a long smooth stretch is still cut so that one
cubic never has to span a whole bay.

### Style

| Option | Default | Meaning |
|---|---|---|
| `--stroke COLOUR` | `#111111` | stroke colour |
| `--stroke-width F` | `1.25` | stroke width in px |
| `--fill {none,palette}` | `none` | outline only, or filled with a generated palette |
| `--hue F` | `12.0` | palette hue in degrees |
| `--sampling {linear,random}` | `linear` | palette sampling |
| `--picking {linear,shuffle}` | `linear` | palette picking |
| `--seed N` | off | seed the palette generator for reproducible colours |
| `--precision N` | `2` | decimal places written to the SVG |

### Reporting

| Option | Default | Meaning |
|---|---|---|
| `--report` / `--no-report` | on | measure the result and print the accuracy block |
| `--report-json PATH` | off | write the full report, including every setting, as JSON |

### Examples

```bat
:: the default: alpha ink, curvature segmentation, black outline
python main.py resources\example-cuba.png

:: choose the output path and the stroke
python main.py ink.png -o ink.svg --stroke "#000000" --stroke-width 1

:: a black-on-white scan: alpha is flat, so use luminance
python main.py scan.png --mask luminance --luminance-threshold 160

:: white chalk on a blackboard
python main.py chalk.png --mask luminance --luminance-threshold 100 --invert

:: a faster, coarser fit
python main.py ink.png --segmentation blind --chunk-size 8

:: a finer fit, at the cost of more curves and more time
python main.py ink.png --max-chunk 8

:: filled shapes with a reproducible palette
python main.py ink.png --fill palette --hue 12 --sampling random --seed 7

:: the straight line baseline, for comparing against the bezier output
python main.py ink.png --curve polygon

:: keep the measurements for a script
python main.py ink.png --report-json report.json --quiet
```

### What the SVG looks like

One `<path>` per contour, closed, with the curves joined end to end so `z` does
not add a straight chord across the shape:

```xml
<svg width="4096px" height="3072px" viewBox="0 0 4096 3072" xmlns="http://www.w3.org/2000/svg" version="1.1">
<style type="text/css"><![CDATA[
.outline {fill:none; stroke:#111111; stroke-width:1.25; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke}
]]></style>
<path class="outline" d="M 311.0,1360.0 C 298.02,1460.81 476.42,1277.5 415.0,1389.0 C ... z" />
```

`fill-rule` is not used: every contour is its own outline, so holes are the
inner contours produced by the tracer rather than an even-odd fill.

## Accuracy

`--report` (on by default) answers two separate questions.

**Coverage** compares the original ink mask with the SVG rasterised back to a
mask, using the even-odd rule.

The `(1 px tolerance)` IoU is the headline number: a pixel counts as correct
when it lands within one pixel of the other mask.

`strict IoU` compares filled areas exactly, and it always looks low. That is not
a conversion failure: `cv.findContours` returns the centres of boundary pixels,
so a polygon drawn through them is half a pixel inside the ink on every side,
and a thin stroke loses a noticeable fraction of its area when refilled. The
`ceiling` next to it is the same measurement made from the **exact** raster
contours with no fitting at all. The percentage is how much of that ceiling the
curve fit reached. On `resources/example-cuba.png` the bezier output reaches
95.1% of what the raw raster outline itself can reach.

**Boundary** answers the direct geometric question, in pixels:

- `raster->vector` Ã¢â‚¬â€ for every raster contour point, the distance to the nearest
  point on the fitted curve. This is the one that matters: is the curve on the ink?
- `vector->raster` Ã¢â‚¬â€ the reverse, which catches a curve bulging away from the ink.
- `hausdorff` Ã¢â‚¬â€ the worst of the two, i.e. the single worst spot in the drawing.
- `within 1 px` / `within 2 px` Ã¢â‚¬â€ the share of raster points that close to the curve.

`compress` is the point of vectorising at all: how many raster points became how
many curves, and how the filled area compares.

Region mode uses the same report, but with the exact pixel-lattice region loops
as `raster_contours` rather than the land silhouette. That measures the fit of
every shared border, not just the outer coastline. Coverage is still compared
with `labels > 0`. The `strict/ceiling` ratio is clamped to 100% because
`cv.fillPoly` treats boundary pixels inclusively and can make the exact-loop
baseline fractionally smaller than the region fill.

### Bezier versus polygon, measured

Running the same map both ways, with the same mask:

| | bezier | polygon |
|---|---|---|
| curves | 3157 | 30267 |
| SVG size | 157 KB | 414 KB |
| fit time | 56 s | 0.04 s |
| coverage IoU (1 px) | 0.9655 | 0.9982 |
| strict IoU / ceiling | 0.6651 / 0.6995 (95.1%) | 0.6995 / 0.6995 (100%) |
| boundary mean | 0.07 px | 0.00 px |
| boundary max | 2.87 px | 0.00 px |
| within 1 px | 99.9% | 100% |

The polygon output is the exact raster outline, so it wins every accuracy number
by construction. The bezier output is 9.6x smaller in points and 2.6x smaller on
disk, and 99.9% of the ink boundary is still within one pixel of the curve. That
is the trade the fitter is making, and the report is what makes it visible.

## Accuracy tool

The same measurements are available on any SVG, without running the converter,
which makes it usable as a regression check or to score output produced
elsewhere (including the older `saveCons()` polygon writer):

```bat
python -m chunkmap.accuracy resources\example-cuba.png outputs\cuba-bezier.svg
python -m chunkmap.accuracy resources\example-cuba.png outputs\cuba-bezier.svg --json report.json
python -m chunkmap.accuracy resources\example-cuba.png resources\cuba_ex.svg
```

It accepts the same `--mask`, `--alpha-threshold`, `--luminance-threshold` and
`--invert` options as the converter, plus `--json PATH` and `--quiet`.

Supported path commands are `M`, `L`, `H`, `V`, `C` and `Z`, absolute or
relative. Anything else is reported as an error rather than silently skipped.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | SVG written (or batch completed above thresholds) |
| `1` | batch only: a conversion failure or a below-threshold image |
| `2` | bad input/config: file missing, undecodable, no ink found, wrong channel count, bad config |
| `130` | interrupted |

## Tests

The suite is split into unit tests (one module at a time) and integration tests
(the real CLI and batch runner):

```bat
venv\Scripts\activate
python -m pytest -q                       :: everything
python -m pytest tests\unit -q            :: one module at a time
python -m pytest tests\integration -q     :: real CLI, generate, regions, orchestrate
```

`pytest.ini` already points at `tests/` and disables the pytest cache, so a bare
`python -m pytest` works too. The suite covers the mask modes, the segmentation
limits, the degenerate cases that used to crash the fitter, the SVG parser and
writer, the palettes, the end to end CLI, the random island generator
(outline/fill, multiple islands, subdivisions), the region-aware shared-border
vectorizer, and the orchestration pipeline (YAML/JSON config, case planning,
overlay rendering, accuracy summaries, failure handling).

The tests write only inside `outputs/_test/` (gitignored) and delete each
scratch directory afterwards, so they also work when the user profile is
read-only.

## Files

| Path | Role |
|---|---|
| `main.py` | entry point; `--help`, conversion, `generate`, `orchestrate`, `--regions` |
| `chunkmap/cli.py` | argument parser and dispatch, shared by `main.py` and the tests |
| `chunkmap/island_generator.py` | random island PNG generator: outline/fill, multiple islands, area-aware subdivisions, label maps |
| `chunkmap/orchestration.py` | integrated batch runner: config -> islands -> SVG -> overlay -> accuracy summary |
| `chunkmap/region_vectorization.py` | label map -> planar region graph -> shared-border filled SVG paths |
| `chunkmap/accuracy.py` | coverage and boundary metrics, plus the standalone scorer |
| `chunkmap/vectorization.py` | bezier fitting (`fitBezierCurve`), segmentation helpers, SVG control point formatting |
| `chunkmap/segmentation.py` | curvature aware and multiscale contour segmentation |
| `chunkmap/fileutils.py` | `SVGHandler`, output folders, path helpers |
| `chunkmap/colorspace.py` | HSL palettes (`cmapHex`, `populatePalette`) |
| `chunkmap/cvutils.py` | alpha histogram, contour reshaping |
| `chunkmap/experimental/` | research fitters, plotting and the original scratchpad; not used by the CLI |
| `tests/unit/` | unit tests for each operational module |
| `tests/integration/` | end-to-end CLI, generate, region and orchestration tests |
| `tests/conftest.py`, `tests/helpers.py` | scratch-dir fixture, matplotlib cache, shared line-art helpers |
| `configs/island-svg-sample.yaml` | sample orchestration config (YAML) |
| `configs/island-svg-sample.json` | the same config in JSON |
| `scripts/help-*.bat` | echo-only help pages for setup, convert, regions, generate, orchestrate, tests, accuracy |
| `help.bat` | menu that opens those help pages |
| `web-app/app.py` | Flask mock server, endpoints are still placeholders |
| `AGENTS.md` | architecture/refactor contract for agents and maintainers |
| `outline.md` | current structure, status and near-term roadmap |

## Using the modules directly

The CLI is a thin layer; the pipeline can still be driven from Python.

```python
import cv2 as cv

from chunkmap import accuracy
from chunkmap import colorspace as colutils
from chunkmap import cvutils as cutils
from chunkmap import fileutils as futils
from chunkmap import vectorization as vect

img = cv.imread('resources/example-cuba.png', cv.IMREAD_UNCHANGED)

# alpha channel -> binary mask
opac, hist = cutils.histimg(img)
ret, thresh = cv.threshold(opac, 252, 255, cv.THRESH_BINARY)

# contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
contours = cutils.contoursRehsaper(contours)

# one colour per contour, then a filled polygon SVG
cmap = colutils.cmapHex(12, len(contours), 'random', 'shuffle')
futils.saveCons(img.shape, contours, cmap, 'outputs', 'cuba_ex')
```

Note that `cvutils.histimg` requires a genuine 4 channel RGBA image and raises a
clear error otherwise; use `accuracy.prepare_mask` if you want the luminance
fallback too.

The region-aware path is similarly small; the generator supplies the label map:

```python
from chunkmap import island_generator as islands
from chunkmap import region_vectorization as regions

result = islands.generate_island(
    islands.IslandParams(1024, 768, seed=42, islands=3, subdivisions=3))

layout = regions.build_layout(result.labels, max_chunk=10, corner_angle_deg=55)
svg = regions.build_svg(layout, colors=regions.default_palette(len(layout.paths)))
with open('island-regions.svg', 'w', encoding='utf-8') as handle:
    handle.write(svg)
```

`layout.paths` contains one region path per label, and shared chains are the same
cubic segments before and after reversal. `layout.exact_loops` is the unfitted
lattice reference used by the accuracy report.

## Acknowledgements

<p>

- Vectorization algorithm: Heavily inspired by Dr. Tim A. Pastva's thesis "<a href = "https://calhoun.nps.edu/entities/publication/8126011c-a7ec-4cad-8372-4c971bf915a9">Bezier Curve Fitting</a>". <b>I have introduced some optimizations for his algorithms, for faster vectorization.</b>
    - Such as cachehing bernstein matrices to save compute on recalculating bernstien matrices each iteration.
    - And early termination techniques for results that achieve a certain level of tolerance.
- Curvature aware adaptive polygon segmentation: A multiple sources were used
    - Wikipedia's equations for <a href="https://en.wikipedia.org/wiki/Curvature#Plane_curves">planar curvature</a> were used to calculate the curvature of a raster curve in 2D plane.
    - The mulit-scalar curvature based segmentation was implemented from F. Mokhtarin's paper "<a href="https://ieeexplore.ieee.org/document/149591">A theory of multiscale, curvature-based shape representation for planar curves</a>".
</p>

## TODO

<p>
This project still very much WIP. There is plenty of more to add.
</p>

<ol start=1>
<li>On-Demand colormap: Spits out colors based on a hue value infinetely.</li>
<li>Easy color editing for an already saved SVG.</li>
<li>Rudimentary UI features.</li>
<li>Write SVG files compatible with inkscape layers.</li>
<li>On-click contour selection in-UI</li>
</ol>

## Known limitations

- Curves are fitted one chain at a time, so two region borders meet at a junction
  but are not guaranteed to be tangent-continuous across it.
- The whole document is one flat group of paths, with no layers or grouping.
  Region paths use `fill-rule="evenodd"` for holes; the ink-band writer still
  emits one independent path per contour.
- Region mode needs a label map. It deliberately does not infer region topology
  from a stroke band: a line could be a coastline, a shared border, or decoration.
  The generator supplies the label map, so `orchestrate` is exact.
- Ink-band mode still traces both sides of a drawn stroke. That is correct for
  hand-drawn line art but wrong for a subdivision map; use `--regions` there.
- Fitting is the slow part: a 4096x3072 map takes between one and a few minutes
  depending on the machine, because each curve runs its own optimisation loop.
  `--segmentation blind` and a larger `--chunk-size` are the fast escape hatches.
- Several legacy helpers in `fileutils.py` still resolve paths against the
  current working directory.

## References

<a href="https://www.w3.org/TR/SVG2/">SVG file definitions</a><br>
<a href="https://en.wikipedia.org/wiki/HSL_and_HSV">HSL theory and HSl to RGB conversions</a>






