"""Integration tests for ink-band conversion through the real CLI."""

import json
import os
import xml.etree.ElementTree as ElementTree

import cv2 as cv
import numpy as np

from chunkmap import accuracy
from chunkmap import cli
from chunkmap import segmentation as seg
from chunkmap import vectorization as vect
from tests.helpers import make_lineart, make_mask


def test_straight_lines_do_not_crash_the_fitter(work):
    """A collinear run makes the covariance matrix singular (used to raise LinAlgError)."""
    mask = make_mask(make_lineart(os.path.join(work, "straight.png")))
    contours, _ = accuracy.trace_contours(mask)
    assert contours

    fits = 0
    for contour in contours:
        points = contour.astype(np.float64)
        for segment in seg.polygon_curvature_segmentation(points, min_chunk=5,
                                                          max_chunk=15):
            control = vect.fitBezierCurve(segment)
            assert control.shape == (4, 2)
            assert np.all(np.isfinite(control))
            fits += 1
    assert fits > 0


def test_polygon_mode_reproduces_the_traced_contours(work):
    """Polygon mode writes the raster points verbatim, so it is the exact reference."""
    source = make_lineart(os.path.join(work, "poly.png"))
    output = os.path.join(work, "poly.svg")
    assert cli.main([source, "--curve", "polygon", "-o", output, "--quiet"]) == 0

    with open(output, encoding="utf-8") as handle:
        document = accuracy.parse_svg(handle.read())
    mask = make_mask(source)
    reconstruction = accuracy.loops_to_mask(document["loops"], mask.shape)

    contours, _ = accuracy.trace_contours(mask)
    baseline = accuracy.loops_to_mask([c.astype(float) for c in contours],
                                      mask.shape)
    assert np.array_equal(reconstruction, baseline)


def test_cli_bezier_end_to_end(work):
    source = make_lineart(os.path.join(work, "e2e.png"))
    output = os.path.join(work, "e2e.svg")
    report = os.path.join(work, "e2e.json")

    code = cli.main([source, "-o", output, "--report-json", report, "--quiet"])
    assert code == 0
    assert os.path.isfile(output)

    with open(output, encoding="utf-8") as handle:
        text = handle.read()

    # the document has to be valid XML, and carry a viewBox with the right casing
    root = ElementTree.fromstring(text)
    assert root.tag.endswith("svg")
    assert root.get("viewBox") == "0 0 240 180"
    assert root.get("width") == "240px"
    assert "C " in text and "z" in text

    with open(report, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["vector"]["curve"] == "bezier"
    assert payload["vector"]["segments"] > 0
    metrics = payload["accuracy"]
    assert metrics["boundary"]["within_1px_pct"] > 90.0
    assert metrics["coverage"]["tolerant"]["iou"] > 0.9
    assert 0.0 <= metrics["relative_to_exact_contours"]["strict_iou"] <= 1.01


def test_cli_reports_a_missing_file():
    assert cli.main(["does-not-exist.png", "--quiet"]) == 2


def test_cli_reports_an_empty_mask(work):
    path = os.path.join(work, "blank.png")
    cv.imwrite(path, np.zeros((40, 40, 4), np.uint8))
    assert cli.main([path, "--quiet"]) == 2


def test_cli_reports_alpha_mode_on_rgb(work):
    path = make_lineart(os.path.join(work, "rgb2.png"), alpha=False)
    assert cli.main([path, "--mask", "alpha", "--quiet"]) == 2
