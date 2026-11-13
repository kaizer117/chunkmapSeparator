# Project outline

_Last updated: 2026-11-12. This file describes what the repository contains now,
where each kind of file lives, and what is still missing. The user manual is
`README.md`; the refactor contract is `AGENTS.md`; this is the map._

## 1. What the project is

Chunk Map Separator turns raster map drawings into clean SVG geometry and
measures how faithful the conversion was. Two vectorisation contracts exist and
must stay separate:

| Mode | Input | Output |
|---|---|---|
| **Ink-band** | PNG whose ink is in alpha, or a luminance fallback | one stroked/closed path per traced contour |
| **Region-aware** | integer label map (0 = water, positive = region) | one filled path per region, shared borders fit once |

A deterministic island generator supplies test maps and label maps, and
`orchestrate` runs the whole loop as a batch integration test.

## 2. Current structure

```
chunkMapSeparator/
  main.py                     root entry point; run everything through this
  chunkmap/                   operational code (importable package)
    __init__.py
    __main__.py               python -m chunkmap alias
    cli.py                    argument parser + dispatch for every mode
    accuracy.py               coverage / boundary metrics, standalone scorer
    vectorization.py          cubic bezier fitting + control-point formatting
    segmentation.py           curvature-aware and blind contour splitting
    region_vectorization.py   label map -> planar graph -> shared borders
    island_generator.py       deterministic islands + region label maps
    orchestration.py          config loader, batch runner, overlay, reports
    fileutils.py              SVG document writer (SVGHandler)
    colorspace.py             HSL palette generation
    cvutils.py                alpha histogram, contour reshaping
    programutils.py           small numeric helpers
    experimental/             research fitters, plots, original scratchpad
  tests/
    conftest.py               scratch dir, sys.path, matplotlib cache
    helpers.py                shared line-art fixtures
    unit/                     61 tests, one module at a time
    integration/              16 tests, real CLI and batch runner
  configs/                    island-svg-sample.yaml / .json
  scripts/                    echo-only help pages for the README commands
  help.bat                    menu that opens those pages
  resources/                  example images and reference SVGs
  research/                   design notes and diagrams
  web-app/                    static browser shell + placeholder Flask mock
  outputs/                    generated SVGs, overlays, reports (gitignored)
  requirements.txt
  pytest.ini
  setup.bat
  README.md  AGENTS.md  outline.md
```

The old flat `code/` folder no longer exists. Its operational modules moved to
`chunkmap/`, its tests split into `tests/unit` and `tests/integration`, its
configs moved to `configs/`, and its research/scratch files moved to
`chunkmap/experimental/`.

## 3. Where to find things

| Task | File |
|---|---|
| Add or change a CLI flag / subcommand | `chunkmap/cli.py`, then `main.py` needs no change |
| Change bezier fitting or segmentation | `chunkmap/vectorization.py`, `chunkmap/segmentation.py` |
| Change SVG output | `chunkmap/fileutils.py`, `chunkmap/region_vectorization.py` |
| Change palette generation | `chunkmap/colorspace.py` |
| Change the region/shared-border algorithm | `chunkmap/region_vectorization.py` |
| Change island generation | `chunkmap/island_generator.py` |
| Change config schema or batch reports | `chunkmap/orchestration.py`, `configs/*` |
| Add a unit test | `tests/unit/test_<module>.py` |
| Add an end-to-end test | `tests/integration/test_cli_*.py` |
| Try an experimental fitter | `chunkmap/experimental/` (not production) |
| Update command help shown to users | `README.md`, then `scripts/help-*.bat` |

## 4. Status

| Area | State | Notes |
|---|---|---|
| Ink-band conversion | Working | bezier + polygon baseline, alpha/luminance masks |
| Accuracy report | Working | tolerant IoU, strict IoU, exact-contour ceiling, boundary metrics |
| Region-aware conversion | Working | shared borders reused in opposite directions; seams hidden |
| Island generator | Working | deterministic per seed; exposes land mask and label map |
| Batch orchestration | Working | balanced cases, overlays, JSON/CSV summaries, failure continuation |
| CLI + docs | Working | `python main.py ...`, four modes, exit codes 0/1/2/130 |
| Tests | Working | 77 passing (61 unit + 16 integration), ~15 s |
| Palette generation | Working, needs review | `hsl2rgb` hue-sector table maps 60°/120°/240° to the wrong channels; left unchanged by this reorg |
| Web UI | Placeholder | Flask endpoints return random mock shapes; nothing is wired to the vectoriser |
| Packaging | Basic | `requirements.txt` + `setup.bat`; no `pyproject.toml`/console script yet |

## 5. Short command reference

```bat
:: setup
setup.bat

:: help pages (echo only, no processing)
help.bat
scripts\help-convert.bat

:: ink-band conversion
python main.py resources\example-cuba.png
python main.py ink.png -o outputs\ink.svg --report-json outputs\ink.json

:: region-aware conversion
python main.py island.png --regions island-labels.png -o island-regions.svg

:: random islands
python main.py generate outputs\islands 8 1024 768 --seed 42

:: batch integration test
python main.py orchestrate --config configs\island-svg-sample.yaml --count 2

:: accuracy of an existing SVG
python -m chunkmap.accuracy resources\example-cuba.png outputs\cuba-bezier.svg

:: tests
python -m pytest -q
```

## 6. Test map

| Command | What it covers |
|---|---|
| `python -m pytest -q` | all 77 tests |
| `python -m pytest tests\unit -q` | 61 module-level tests |
| `python -m pytest tests\integration -q` | 16 CLI/batch tests |
| `python -m pytest tests\unit\test_region_vectorization.py -q` | shared-border geometry |
| `python -m pytest tests\integration\test_orchestration_run.py -q` | full batch pipeline |

All scratch output goes under `outputs/_test/` and is removed per test. The
suite does not need network access and does not modify checked-in resources.

## 7. Help pages

`help.bat` is a menu. Each page below only prints commands and expected output;
it does not run anything.

| Page | Topic |
|---|---|
| `scripts\help-setup.bat` | activate `venv`, install from requirements |
| `scripts\help-convert.bat` | ink-band conversion and expected report |
| `scripts\help-regions.bat` | label-map conversion and expected report |
| `scripts\help-generate.bat` | random island PNG options |
| `scripts\help-orchestrate.bat` | batch run, output folders, summary |
| `scripts\help-tests.bat` | unit / integration / single-test commands |
| `scripts\help-accuracy.bat` | standalone scorer on an existing SVG |

## 8. Suggested next steps

1. Wire `web-app/app.py` to a real `process_lineart()` service: upload,
   vectorise, return SVG, download. The current endpoints are still random.
2. Fix the `hsl2rgb` hue-sector mapping and add a unit test for the six primary
   sectors; this changes palettes, so treat it as a visible product change.
3. Add a `pyproject.toml` with a `chunkmap-separator` console script and pin the
   Python version, so `python main.py` is not the only entry point.
4. Add optional SVG layers/groups for Inkscape compatibility.
5. Decide whether `fileutils.py` should be split into `svg.py` (writer) and
   `paths.py` (legacy folder helpers); today it is one module for compatibility.
