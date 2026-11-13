"""Unit tests for :mod:`chunkmap.colorspace` (palette generation)."""

import re

import numpy as np

from chunkmap import colorspace as colutils

HEX = re.compile(r"^#[0-9a-f]{6}$")


def test_rgb2hex_pads_and_lowercases():
    assert colutils.rgb2hex([1.0, 0.0, 0.0]) == "#ff0000"
    assert colutils.rgb2hex([0.0, 1.0, 0.0]) == "#00ff00"
    assert colutils.rgb2hex([0.0, 0.0, 1.0]) == "#0000ff"
    assert colutils.rgb2hex([0.0, 0.0, 0.0]) == "#000000"


def test_cmap_hex_returns_the_requested_number_of_colours():
    for n in (1, 5, 12, 13, 25):
        colours = colutils.cmapHex(12.0, n)
        assert len(colours) == n
        assert all(HEX.match(colour) for colour in colours)


def test_cmap_hex_is_reproducible_with_a_seed():
    np.random.seed(7)
    first = colutils.cmapHex(120.0, 12, "random", "shuffle")
    np.random.seed(7)
    second = colutils.cmapHex(120.0, 12, "random", "shuffle")
    assert first == second


def test_get_rand_colours_returns_valid_hex():
    colours = colutils.getRandColours(6)
    assert len(colours) == 6
    assert all(HEX.match(colour) for colour in colours)
