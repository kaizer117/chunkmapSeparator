"""Chunk Map Separator — operational package.

Everything that does real work lives here:

* ``cli``                  command-line parsing and dispatch
* ``accuracy``             coverage / boundary measurement
* ``vectorization``        cubic bezier fitting and SVG control-point layout
* ``segmentation``         curvature-aware and blind contour splitting
* ``region_vectorization`` shared-border region fills
* ``island_generator``     deterministic random island / label-map generator
* ``orchestration``        batch runner, config loader and reporting
* ``fileutils``            SVG document writer (``SVGHandler``)
* ``colorspace``           HSL palettes
* ``cvutils``, ``programutils``  small raster / numeric helpers

The research fitters and scratch scripts are isolated in
:mod:`chunkmap.experimental` and are deliberately not imported here; keep the
package import cheap so ``python main.py generate`` does not pull in matplotlib.

Run the tool through the repository entry point::

    python main.py --help
"""

__version__ = "0.2.0"
