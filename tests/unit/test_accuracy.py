"""Unit tests for :mod:`chunkmap.accuracy`.

Covers the mask resolver, the SVG path parser, the even-odd reconstruction and
the mask scorer. The end-to-end accuracy report is exercised in
``tests/integration/test_cli_convert.py``.
"""

import os

import cv2 as cv
import numpy as np
import pytest

from chunkmap import accuracy
from tests.helpers import PROJECT_ROOT, make_lineart, make_mask


# ---------------------------------------------------------------------------
# alpha / luminance mask resolution
# ---------------------------------------------------------------------------

def test_alpha_is_used_when_it_carries_the_art(work):
    image = cv.imread(make_lineart(os.path.join(work, "alpha.png")),
                      cv.IMREAD_UNCHANGED)
    info = accuracy.describe_alpha(image)
    assert info["present"] and info["usable"]

    mask, mode, note = accuracy.prepare_mask(image, "auto")
    assert mode == "alpha"
    assert note == ""
    assert np.count_nonzero(mask) > 0


def test_flat_alpha_falls_back_to_luminance(work):
    """Black lines on an opaque white background: alpha is 255 everywhere."""
    path = make_lineart(os.path.join(work, "white.png"), alpha=False,
                        background=(255, 255, 255))
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    assert accuracy.describe_alpha(image)["present"] is False

    mask, mode, note = accuracy.prepare_mask(image, "auto")
    assert mode == "luminance"
    assert "no alpha channel" in note
    assert np.count_nonzero(mask) > 0


def test_uniform_alpha_reports_itself_as_unusable(work):
    path = make_lineart(os.path.join(work, "flat.png"),
                        background=(255, 255, 255))
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    info = accuracy.describe_alpha(image)
    assert info["present"] and not info["usable"]

    _, mode, note = accuracy.prepare_mask(image, "auto")
    assert mode == "luminance"
    assert "uniform" in note


def test_alpha_mode_rejects_rgb(work):
    path = make_lineart(os.path.join(work, "rgb.png"), alpha=False)
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    with pytest.raises(ValueError):
        accuracy.prepare_mask(image, "alpha")


# ---------------------------------------------------------------------------
# SVG path parsing
# ---------------------------------------------------------------------------

def test_parser_reads_the_checked_in_polygon_svg():
    path = os.path.join(PROJECT_ROOT, "resources", "cuba_ex.svg")
    if not os.path.isfile(path):
        pytest.skip("resources/cuba_ex.svg is missing")
    with open(path, encoding="utf-8") as handle:
        document = accuracy.parse_svg(handle.read())
    assert document["paths"] == 66
    assert len(document["loops"]) == 66
    assert document["width"] == 4096 and document["height"] == 3072


def test_path_parser_handles_relative_commands():
    loops = accuracy.parse_path_data("M 10 10 l 10 0 l 0 10 z")
    assert len(loops) == 1
    assert np.allclose(loops[0].max(axis=0), [20.0, 20.0])


def test_path_parser_rejects_unsupported_commands():
    with pytest.raises(ValueError):
        accuracy.parse_path_data("M 0 0 A 5 5 0 0 1 10 10 Z")


# ---------------------------------------------------------------------------
# reconstruction and scoring
# ---------------------------------------------------------------------------

def test_loops_to_mask_fills_a_closed_loop():
    loop = np.array([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]])
    mask = accuracy.loops_to_mask([loop], (12, 12))
    assert mask[5, 5] == 255
    assert mask[0, 0] == 0
    assert np.count_nonzero(mask) > 0


def test_loops_to_mask_honours_even_odd_holes():
    outer = np.array([[2.0, 2.0], [9.0, 2.0], [9.0, 9.0], [2.0, 9.0]])
    hole = np.array([[4.0, 4.0], [7.0, 4.0], [7.0, 7.0], [4.0, 7.0]])
    mask = accuracy.loops_to_mask([outer, hole], (12, 12))
    assert mask[5, 5] == 0          # hole cancelled by the even-odd rule
    assert mask[3, 3] == 255        # ring between the two loops


def test_mask_scores_perfect_match_is_one():
    reference = np.zeros((20, 20), np.uint8)
    reference[5:15, 5:15] = 255
    scores = accuracy.mask_scores(reference, reference, tolerance=1)
    assert scores["iou"] == 1.0
    assert scores["dice"] == 1.0
    assert scores["tolerant"]["iou"] == 1.0
    assert scores["false_positive_px"] == 0


def test_ctrl_to_loop_flattens_the_formatter_layout():
    rows = [
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
        np.array([[3.0, 1.0], [4.0, 1.0], [5.0, 0.0]]),
    ]
    loop = accuracy.ctrl_to_loop(rows, samples=4)
    assert loop.shape[1] == 2
    assert np.allclose(loop[0], [0.0, 0.0])
    assert np.allclose(loop[-1], [5.0, 0.0])
    assert len(loop) > 2


def test_make_mask_helper_returns_binary_ink(work):
    mask = make_mask(make_lineart(os.path.join(work, "helper.png")))
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask) > 0
